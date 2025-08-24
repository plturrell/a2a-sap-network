sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel",
    "a2a/network/agent9/ext/utils/SecurityUtils",
    "a2a/network/agent9/ext/utils/AuthHandler"
], function (ControllerExtension, MessageBox, MessageToast, Fragment, JSONModel, SecurityUtils, AuthHandler) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent9.ext.controller.ObjectPageExt", {
        
        override: {
            onInit: function () {
                this._extensionAPI = this.base.getExtensionAPI();
                this._securityUtils = SecurityUtils;
                this._authHandler = AuthHandler;
                
                // Initialize device model for responsive behavior
                var oDeviceModel = new JSONModel(sap.ui.Device);
                oDeviceModel.setDefaultBindingMode(sap.ui.model.BindingMode.OneWay);
                this.base.getView().setModel(oDeviceModel, "device");
                
                // Initialize create dialog model
                this._initializeCreateModel();
                
                // Initialize security and audit logging
                this._initializeSecurity();
            },
            
            onExit: function() {
                this._cleanupResources();
                if (this.base.onExit) {
                    this.base.onExit.apply(this, arguments);
                }
            }
        },

        /**
         * @function _initializeSecurity
         * @description Initializes security features and audit logging.
         * @private
         */
        _initializeSecurity: function() {
            this._auditLogger = {
                log: function(action, details) {
                    var user = this._getCurrentUser();
                    var timestamp = new Date().toISOString();
                    var logEntry = {
                        timestamp: timestamp,
                        user: user,
                        agent: "Agent9_APIIntegration",
                        action: action,
                        details: details || {}
                    };
                    console.info("AUDIT: " + JSON.stringify(logEntry));
                }.bind(this)
            };
        },

        /**
         * @function _getCurrentUser
         * @description Gets current user ID for audit logging.
         * @returns {string} User ID or "anonymous"
         * @private
         */
        _getCurrentUser: function() {
            return sap.ushell?.Container?.getUser()?.getId() || "anonymous";
        },

        /**
         * @function _hasRole
         * @description Checks if current user has specified role.
         * @param {string} role - Role to check
         * @returns {boolean} True if user has role
         * @private
         */
        _hasRole: function(role) {
            const user = sap.ushell?.Container?.getUser();
            if (user && user.hasRole) {
                return user.hasRole(role);
            }
            // Mock role validation for development/testing
            const mockRoles = ["IntegrationAdmin", "IntegrationUser", "IntegrationOperator"];
            return mockRoles.includes(role);
        },

        /**
         * @function _cleanupResources
         * @description Cleans up resources to prevent memory leaks.
         * @private
         */
        _cleanupResources: function() {
            // Clean up event sources
            if (this._connectionEventSource) {
                this._connectionEventSource.close();
                this._connectionEventSource = null;
            }
            if (this._integrationEventSource) {
                this._integrationEventSource.close();
                this._integrationEventSource = null;
            }
            
            // Clean up cached dialogs
            var aDialogs = ["_oTestDialog", "_oAuthDialog", "_oLogsDialog", "_oConfigDialog"];
            aDialogs.forEach(function(sDialog) {
                if (this[sDialog]) {
                    this[sDialog].destroy();
                    this[sDialog] = null;
                }
            }.bind(this));
            
            this._auditLogger.log('RESOURCES_CLEANED_UP', 'ObjectPage resources cleaned up on exit');
        },

        _initializeCreateModel: function() {
            var oCreateData = {
                integrationName: "",
                description: "",
                integrationType: "REST_API",
                endpointUrl: "",
                authMethod: "API_KEY",
                protocol: "HTTPS",
                timeout: 30000,
                priority: "MEDIUM",
                // Validation states
                integrationNameState: "None",
                integrationNameStateText: "",
                endpointUrlState: "None",
                endpointUrlStateText: "",
                integrationTypeState: "None",
                integrationTypeStateText: "",
                // Configuration options
                retryPolicy: {
                    maxRetries: 3,
                    retryDelay: 1000,
                    exponentialBackoff: true
                },
                rateLimiting: {
                    requestsPerSecond: 10,
                    burstSize: 50,
                    enabled: true
                },
                dataFormat: "JSON",
                validation: {
                    validateSchema: true,
                    strictMode: true,
                    sanitizeInput: true
                },
                monitoring: {
                    logRequests: true,
                    trackPerformance: true,
                    alertOnFailure: true
                },
                // Available options (loaded dynamically)
                integrationTypes: [],
                authMethods: [],
                protocols: [],
                dataFormats: []
            };
            var oCreateModel = new JSONModel(oCreateData);
            this.base.getView().setModel(oCreateModel, "create");
        },

        /**
         * @function onTestConnection
         * @description Tests API connection for the current integration.
         * @public
         */
        onTestConnection: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sIntegrationId = oContext.getProperty("integrationId");
            var sIntegrationName = oContext.getProperty("integrationName");
            
            // Security check: Validate user authorization
            if (!this._hasRole("IntegrationUser")) {
                MessageBox.error("Access denied: Insufficient privileges for testing connections");
                this._auditLogger.log("TEST_CONNECTION_ACCESS_DENIED", "User attempted connection test without IntegrationUser role");
                return;
            }
            
            // Validate integration ID
            if (this._securityUtils && !this._securityUtils.validateIntegrationId(sIntegrationId)) {
                MessageBox.error("Invalid integration ID");
                this._auditLogger.log("TEST_CONNECTION_INVALID_ID", "Invalid integration ID provided", { integrationId: sIntegrationId });
                return;
            }
            
            const safeIntegrationName = this._securityUtils ? 
                this._securityUtils.encodeHTML(sIntegrationName) : 
                sIntegrationName;
            
            MessageBox.confirm("Test connection for '" + safeIntegrationName + "'?", {
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        this._executeConnectionTest(sIntegrationId, sIntegrationName);
                    }
                }.bind(this)
            });
        },

        /**
         * @function _executeConnectionTest
         * @description Executes connection test with real-time monitoring.
         * @param {string} sIntegrationId - Integration ID to test
         * @param {string} sIntegrationName - Integration name for display
         * @private
         */
        _executeConnectionTest: function(sIntegrationId, sIntegrationName) {
            this._extensionAPI.getView().setBusy(true);
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    jQuery.ajax({
                        url: "/a2a/agent9/v1/integrations/" + encodeURIComponent(sIntegrationId) + "/test",
                        type: "POST",
                        headers: {
                            "X-Requested-With": "XMLHttpRequest",
                            "X-CSRF-Token": this._getCSRFToken()
                        },
                        timeout: 60000, // 1 minute timeout for connection tests
                        success: function(data) {
                            resolve(data);
                        },
                        error: function(xhr, textStatus, errorThrown) {
                            var errorMsg = "Network error";
                            if (xhr.responseText) {
                                errorMsg = this._securityUtils ? 
                                    this._securityUtils.sanitizeErrorMessage(xhr.responseText) : 
                                    "Connection test failed";
                            }
                            reject(new Error(errorMsg));
                        }.bind(this)
                    });
                }.bind(this));
            }.bind(this)).then(function(data) {
                this._extensionAPI.getView().setBusy(false);
                this._showConnectionTestResults(data, sIntegrationName);
                this._auditLogger.log("TEST_CONNECTION_SUCCESS", "Connection test successful", {
                    integrationId: sIntegrationId,
                    integrationName: sIntegrationName,
                    responseTime: data.responseTimeMs
                });
            }.bind(this)).catch(function(error) {
                this._extensionAPI.getView().setBusy(false);
                MessageBox.error("Connection test failed: " + error.message);
                this._auditLogger.log("TEST_CONNECTION_FAILED", "Connection test failed", {
                    integrationId: sIntegrationId,
                    error: error.message
                });
            }.bind(this));
        },

        /**
         * @function _showConnectionTestResults
         * @description Displays connection test results in a formatted dialog.
         * @param {Object} testData - Connection test results
         * @param {string} integrationName - Integration name for display
         * @private
         */
        _showConnectionTestResults: function(testData, integrationName) {
            const safeIntegrationName = this._securityUtils ? 
                this._securityUtils.encodeHTML(integrationName) : 
                integrationName;
            const safeStatus = this._securityUtils ? 
                this._securityUtils.encodeHTML(testData.status) : 
                testData.status;
            const safeResponseTime = parseInt(testData.responseTimeMs) || 0;
            const safeEndpoint = this._securityUtils ? 
                this._securityUtils.encodeHTML(testData.endpoint) : 
                testData.endpoint;
            
            var sMessage = "Connection Test Results for '" + safeIntegrationName + "':\n\n";
            sMessage += "Status: " + safeStatus + "\n";
            sMessage += "Response Time: " + safeResponseTime + " ms\n";
            sMessage += "Endpoint: " + safeEndpoint + "\n";
            
            if (testData.httpStatusCode) {
                sMessage += "HTTP Status: " + parseInt(testData.httpStatusCode) + "\n";
            }
            
            if (testData.sslVerified !== undefined) {
                sMessage += "SSL Verified: " + (testData.sslVerified ? "Yes" : "No") + "\n";
            }
            
            if (testData.authenticationStatus) {
                const safeAuthStatus = this._securityUtils ? 
                    this._securityUtils.encodeHTML(testData.authenticationStatus) : 
                    testData.authenticationStatus;
                sMessage += "Authentication: " + safeAuthStatus + "\n";
            }
            
            if (testData.rateLimitInfo && testData.rateLimitInfo.remaining !== undefined) {
                sMessage += "Rate Limit Remaining: " + parseInt(testData.rateLimitInfo.remaining) + "\n";
            }
            
            if (testData.errorDetails && !testData.success) {
                sMessage += "\nError Details:\n";
                const safeErrorDetails = this._securityUtils ? 
                    this._securityUtils.encodeHTML(testData.errorDetails) : 
                    testData.errorDetails;
                sMessage += safeErrorDetails;
            }
            
            var sIcon = testData.success ? MessageBox.Icon.SUCCESS : MessageBox.Icon.ERROR;
            MessageBox.show(sMessage, {
                icon: sIcon,
                title: "Connection Test Results"
            });
        },

        /**
         * @function onRunIntegration
         * @description Runs the API integration with the configured parameters.
         * @public
         */
        onRunIntegration: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sIntegrationId = oContext.getProperty("integrationId");
            var sIntegrationName = oContext.getProperty("integrationName");
            var sStatus = oContext.getProperty("status");
            
            // Security check: Validate user authorization
            if (!this._hasRole("IntegrationOperator")) {
                MessageBox.error("Access denied: Insufficient privileges for running integrations");
                this._auditLogger.log("RUN_INTEGRATION_ACCESS_DENIED", "User attempted to run integration without IntegrationOperator role");
                return;
            }
            
            // Validate integration status
            if (sStatus !== "CONFIGURED" && sStatus !== "TESTED") {
                MessageBox.warning("Integration must be configured and tested before running");
                return;
            }
            
            const safeIntegrationName = this._securityUtils ? 
                this._securityUtils.encodeHTML(sIntegrationName) : 
                sIntegrationName;
            
            MessageBox.confirm("Run integration '" + safeIntegrationName + "'?", {
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        this._executeIntegration(sIntegrationId, sIntegrationName);
                    }
                }.bind(this)
            });
        },

        /**
         * @function _executeIntegration
         * @description Executes the integration with real-time monitoring.
         * @param {string} sIntegrationId - Integration ID to run
         * @param {string} sIntegrationName - Integration name for display
         * @private
         */
        _executeIntegration: function(sIntegrationId, sIntegrationName) {
            this._extensionAPI.getView().setBusy(true);
            
            // Start real-time monitoring for integration execution
            this._startIntegrationMonitoring(sIntegrationId);
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    jQuery.ajax({
                        url: "/a2a/agent9/v1/integrations/" + encodeURIComponent(sIntegrationId) + "/run",
                        type: "POST",
                        headers: {
                            "X-Requested-With": "XMLHttpRequest",
                            "X-CSRF-Token": this._getCSRFToken()
                        },
                        timeout: 300000, // 5 minutes timeout for integration execution
                        success: function(data) {
                            resolve(data);
                        },
                        error: function(xhr) {
                            var errorMsg = this._securityUtils ? 
                                this._securityUtils.sanitizeErrorMessage(xhr.responseText) : 
                                "Integration execution failed";
                            reject(new Error(errorMsg));
                        }.bind(this)
                    });
                }.bind(this));
            }.bind(this)).then(function(data) {
                this._extensionAPI.getView().setBusy(false);
                this._showIntegrationResults(data, sIntegrationName);
                this._extensionAPI.refresh();
                this._auditLogger.log("RUN_INTEGRATION_SUCCESS", "Integration executed successfully", {
                    integrationId: sIntegrationId,
                    integrationName: sIntegrationName,
                    executionTime: data.executionTimeMs
                });
            }.bind(this)).catch(function(error) {
                this._extensionAPI.getView().setBusy(false);
                MessageBox.error("Integration execution failed: " + error.message);
                this._auditLogger.log("RUN_INTEGRATION_FAILED", "Integration execution failed", {
                    integrationId: sIntegrationId,
                    error: error.message
                });
            }.bind(this));
        },

        /**
         * @function _startIntegrationMonitoring
         * @description Starts real-time monitoring for integration execution.
         * @param {string} sIntegrationId - Integration ID being executed
         * @private
         */
        _startIntegrationMonitoring: function(sIntegrationId) {
            if (this._integrationEventSource) {
                this._integrationEventSource.close();
            }
            
            var monitoringUrl = "/a2a/agent9/v1/integrations/" + encodeURIComponent(sIntegrationId) + "/monitor";
            if (this._securityUtils && !this._securityUtils.validateEventSourceUrl(monitoringUrl)) {
                console.warn("Invalid integration monitoring URL, skipping real-time monitoring");
                return;
            }
            
            this._integrationEventSource = new EventSource(monitoringUrl);
            
            this._integrationEventSource.onmessage = function(event) {
                try {
                    var data = JSON.parse(event.data);
                    this._updateIntegrationProgress(data);
                } catch (error) {
                    console.error("Error processing integration update:", error);
                }
            }.bind(this);
            
            this._integrationEventSource.onerror = function() {
                var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                var sMessage = oBundle.getText("error.integrationMonitoringLost") || "Integration monitoring connection lost";
                MessageToast.show(sMessage);
            }.bind(this);
        },

        /**
         * @function _updateIntegrationProgress
         * @description Updates integration progress display with real-time data.
         * @param {Object} progressData - Real-time integration progress data
         * @private
         */
        _updateIntegrationProgress: function(progressData) {
            if (progressData.type === "progress_update") {
                const safeStage = this._securityUtils ? 
                    this._securityUtils.encodeHTML(progressData.stage) : 
                    progressData.stage;
                const safeProgress = parseInt(progressData.progress) || 0;
                const safeRecordsProcessed = parseInt(progressData.recordsProcessed) || 0;
                
                var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                var sMessage = oBundle.getText("msg.integrationProgress") || "Integration progress";
                sMessage += ": " + safeStage + " (" + safeProgress + "%)";
                if (safeRecordsProcessed > 0) {
                    sMessage += " - " + safeRecordsProcessed + " records processed";
                }
                MessageToast.show(sMessage);
            } else if (progressData.type === "error") {
                const safeError = this._securityUtils ? 
                    this._securityUtils.sanitizeErrorMessage(progressData.error) : 
                    progressData.error;
                MessageToast.show("Integration Error: " + safeError);
            }
        },

        /**
         * @function onViewLogs
         * @description Opens the logs viewer for the current integration.
         * @public
         */
        onViewLogs: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sIntegrationId = oContext.getProperty("integrationId");
            var sIntegrationName = oContext.getProperty("integrationName");
            
            // Security check: Validate user authorization
            if (!this._hasRole("IntegrationUser")) {
                MessageBox.error("Access denied: Insufficient privileges for viewing logs");
                this._auditLogger.log("VIEW_LOGS_ACCESS_DENIED", "User attempted to view logs without IntegrationUser role");
                return;
            }
            
            this._getOrCreateLogsDialog().then(function(oDialog) {
                var oModel = new JSONModel({
                    integrationId: sIntegrationId,
                    integrationName: sIntegrationName,
                    logs: [],
                    logLevel: "INFO",
                    dateRange: {
                        from: new Date(Date.now() - 24 * 60 * 60 * 1000), // Last 24 hours
                        to: new Date()
                    },
                    autoRefresh: false,
                    maxLines: 1000
                });
                oDialog.setModel(oModel, "logs");
                oDialog.open();
                this._loadIntegrationLogs(sIntegrationId, oDialog);
                
                this._auditLogger.log("VIEW_LOGS_OPENED", "Logs viewer opened", {
                    integrationId: sIntegrationId,
                    integrationName: sIntegrationName
                });
            }.bind(this));
        },

        /**
         * @function onConfigureAuth
         * @description Opens authentication configuration dialog for the current integration.
         * @public
         */
        onConfigureAuth: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sIntegrationId = oContext.getProperty("integrationId");
            var sIntegrationName = oContext.getProperty("integrationName");
            var sCurrentAuthMethod = oContext.getProperty("authMethod");
            
            // Security check: Validate user authorization
            if (!this._hasRole("IntegrationAdmin")) {
                MessageBox.error("Access denied: Insufficient privileges for configuring authentication");
                this._auditLogger.log("CONFIGURE_AUTH_ACCESS_DENIED", "User attempted to configure auth without IntegrationAdmin role");
                return;
            }
            
            this._getOrCreateAuthDialog().then(function(oDialog) {
                var oModel = new JSONModel({
                    integrationId: sIntegrationId,
                    integrationName: sIntegrationName,
                    authMethod: sCurrentAuthMethod || "API_KEY",
                    apiKey: "",
                    apiSecret: "",
                    bearerToken: "",
                    basicAuth: {
                        username: "",
                        password: ""
                    },
                    oauth2: {
                        clientId: "",
                        clientSecret: "",
                        tokenUrl: "",
                        scope: ""
                    },
                    customHeaders: [],
                    availableAuthMethods: [
                        { key: "API_KEY", text: "API Key" },
                        { key: "BEARER_TOKEN", text: "Bearer Token" },
                        { key: "BASIC_AUTH", text: "Basic Authentication" },
                        { key: "OAUTH2", text: "OAuth 2.0" },
                        { key: "CUSTOM", text: "Custom Headers" }
                    ]
                });
                oDialog.setModel(oModel, "auth");
                oDialog.open();
                
                this._auditLogger.log("CONFIGURE_AUTH_OPENED", "Authentication configuration dialog opened", {
                    integrationId: sIntegrationId,
                    integrationName: sIntegrationName
                });
            }.bind(this));
        },

        /**
         * @function _getOrCreateLogsDialog
         * @description Gets or creates the logs viewer dialog.
         * @returns {Promise} Promise resolving to the logs dialog
         * @private
         */
        _getOrCreateLogsDialog: function() {
            if (this._oLogsDialog) {
                return Promise.resolve(this._oLogsDialog);
            }
            
            return Fragment.load({
                id: this.base.getView().getId(),
                name: "a2a.network.agent9.ext.fragment.LogsViewer",
                controller: this
            }).then(function(oDialog) {
                this._oLogsDialog = oDialog;
                this.base.getView().addDependent(this._oLogsDialog);
                return oDialog;
            }.bind(this));
        },

        /**
         * @function _getOrCreateAuthDialog
         * @description Gets or creates the authentication configuration dialog.
         * @returns {Promise} Promise resolving to the auth dialog
         * @private
         */
        _getOrCreateAuthDialog: function() {
            if (this._oAuthDialog) {
                return Promise.resolve(this._oAuthDialog);
            }
            
            return Fragment.load({
                id: this.base.getView().getId(),
                name: "a2a.network.agent9.ext.fragment.AuthConfiguration",
                controller: this
            }).then(function(oDialog) {
                this._oAuthDialog = oDialog;
                this.base.getView().addDependent(this._oAuthDialog);
                return oDialog;
            }.bind(this));
        },

        /**
         * @function _loadIntegrationLogs
         * @description Loads integration logs with filtering and pagination.
         * @param {string} sIntegrationId - Integration ID
         * @param {sap.m.Dialog} oDialog - Logs dialog
         * @private
         */
        _loadIntegrationLogs: function(sIntegrationId, oDialog) {
            var oModel = oDialog.getModel("logs");
            var oData = oModel.getData();
            
            oDialog.setBusy(true);
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    jQuery.ajax({
                        url: "/a2a/agent9/v1/integrations/" + encodeURIComponent(sIntegrationId) + "/logs",
                        type: "GET",
                        data: {
                            level: oData.logLevel,
                            from: oData.dateRange.from.toISOString(),
                            to: oData.dateRange.to.toISOString(),
                            maxLines: oData.maxLines
                        },
                        success: function(data) {
                            resolve(data);
                        },
                        error: function(xhr) {
                            var errorMsg = this._securityUtils ? 
                                this._securityUtils.sanitizeErrorMessage(xhr.responseText) : 
                                "Failed to load logs";
                            reject(new Error(errorMsg));
                        }.bind(this)
                    });
                }.bind(this));
            }.bind(this)).then(function(data) {
                oDialog.setBusy(false);
                oData.logs = this._sanitizeLogEntries(data.logs || []);
                oModel.setData(oData);
                this._auditLogger.log("LOGS_LOADED", "Integration logs loaded", {
                    integrationId: sIntegrationId,
                    logCount: data.logs.length
                });
            }.bind(this)).catch(function(error) {
                oDialog.setBusy(false);
                MessageBox.error("Failed to load logs: " + error.message);
                this._auditLogger.log("LOGS_LOAD_FAILED", "Failed to load integration logs", {
                    integrationId: sIntegrationId,
                    error: error.message
                });
            }.bind(this));
        },


        /**
         * @function _sanitizeLogEntries
         * @description Sanitizes log entries for security display.
         * @param {Array} aLogs - Array of log entries
         * @returns {Array} Sanitized log entries
         * @private
         */
        _sanitizeLogEntries: function(aLogs) {
            if (!Array.isArray(aLogs)) return [];
            
            return aLogs.map(function(logEntry) {
                return {
                    timestamp: logEntry.timestamp,
                    level: this._securityUtils ? this._securityUtils.encodeHTML(logEntry.level) : logEntry.level,
                    message: this._securityUtils ? this._securityUtils.encodeHTML(logEntry.message) : logEntry.message,
                    component: this._securityUtils ? this._securityUtils.encodeHTML(logEntry.component) : logEntry.component,
                    requestId: this._securityUtils ? this._securityUtils.encodeHTML(logEntry.requestId) : logEntry.requestId,
                    duration: parseInt(logEntry.duration) || 0,
                    statusCode: parseInt(logEntry.statusCode) || 0
                };
            }.bind(this));
        },

        /**
         * @function _showIntegrationResults
         * @description Displays integration execution results.
         * @param {Object} resultData - Integration execution results
         * @param {string} integrationName - Integration name for display
         * @private
         */
        _showIntegrationResults: function(resultData, integrationName) {
            const safeIntegrationName = this._securityUtils ? 
                this._securityUtils.encodeHTML(integrationName) : 
                integrationName;
            const safeStatus = this._securityUtils ? 
                this._securityUtils.encodeHTML(resultData.status) : 
                resultData.status;
            const safeRecordsProcessed = parseInt(resultData.recordsProcessed) || 0;
            const safeRecordsSuccessful = parseInt(resultData.recordsSuccessful) || 0;
            const safeRecordsFailed = parseInt(resultData.recordsFailed) || 0;
            const safeExecutionTime = parseInt(resultData.executionTimeMs) || 0;
            
            var sMessage = "Integration Execution Results for '" + safeIntegrationName + "':\n\n";
            sMessage += "Status: " + safeStatus + "\n";
            sMessage += "Execution Time: " + safeExecutionTime + " ms\n";
            sMessage += "Records Processed: " + safeRecordsProcessed + "\n";
            sMessage += "Records Successful: " + safeRecordsSuccessful + "\n";
            sMessage += "Records Failed: " + safeRecordsFailed + "\n";
            
            if (resultData.successRate !== undefined) {
                const safeSuccessRate = parseFloat(resultData.successRate) || 0;
                sMessage += "Success Rate: " + safeSuccessRate + "%\n";
            }
            
            if (resultData.throughput !== undefined) {
                const safeThroughput = parseFloat(resultData.throughput) || 0;
                sMessage += "Throughput: " + safeThroughput + " records/sec\n";
            }
            
            if (resultData.errorSummary && resultData.errorSummary.length > 0) {
                sMessage += "\nError Summary:\n";
                resultData.errorSummary.forEach(function(error) {
                    const safeErrorType = this._securityUtils ? 
                        this._securityUtils.encodeHTML(error.type) : 
                        error.type;
                    const safeErrorCount = parseInt(error.count) || 0;
                    sMessage += "• " + safeErrorType + ": " + safeErrorCount + " occurrences\n";
                }.bind(this));
            }
            
            var sIcon = resultData.status === "COMPLETED" ? MessageBox.Icon.SUCCESS : MessageBox.Icon.WARNING;
            MessageBox.show(sMessage, {
                icon: sIcon,
                title: "Integration Results"
            });
        },

        /**
         * @function _withErrorRecovery
         * @description Wraps operation with error recovery and retry logic.
         * @param {Function} fnOperation - Operation to execute
         * @param {Object} oOptions - Recovery options (optional)
         * @returns {Promise} Promise with error recovery
         * @private
         */
        _withErrorRecovery: function(fnOperation, oOptions) {
            var oConfig = Object.assign({
                maxRetries: 3,
                retryDelay: 1000,
                exponentialBackoff: true
            }, oOptions);
            
            function attempt(retriesLeft, delay) {
                return fnOperation().catch(function(error) {
                    if (retriesLeft > 0) {
                        var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                        var sRetryMsg = oBundle.getText("recovery.retrying") || "Network error. Retrying...";
                        MessageToast.show(sRetryMsg);
                        
                        return new Promise(function(resolve) {
                            setTimeout(resolve, delay);
                        }).then(function() {
                            var nextDelay = oConfig.exponentialBackoff ? delay * 2 : delay;
                            return attempt.call(this, retriesLeft - 1, nextDelay);
                        }.bind(this));
                    }
                    throw error;
                }.bind(this));
            }
            
            return attempt.call(this, oConfig.maxRetries, oConfig.retryDelay);
        },

        /**
         * @function _getCSRFToken
         * @description Gets CSRF token for secure POST requests.
         * @returns {string} CSRF token
         * @private
         */
        _getCSRFToken: function() {
            return this._securityUtils ? this._securityUtils.getCSRFToken() : "placeholder-token";
        },

        /**
         * @function onConfirmAuthConfiguration
         * @description Confirms and saves authentication configuration.
         * @public
         */
        onConfirmAuthConfiguration: function() {
            var oModel = this._oAuthDialog.getModel("auth");
            var oData = oModel.getData();
            
            // Validate authentication configuration
            if (!this._validateAuthConfiguration(oData)) {
                return;
            }
            
            // Sanitize and prepare data for secure transmission
            var oAuthData = this._sanitizeAuthConfiguration(oData);
            
            this._oAuthDialog.setBusy(true);
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    jQuery.ajax({
                        url: "/a2a/agent9/v1/integrations/" + encodeURIComponent(oData.integrationId) + "/auth",
                        type: "PUT",
                        contentType: "application/json",
                        headers: {
                            "X-Requested-With": "XMLHttpRequest",
                            "X-CSRF-Token": this._getCSRFToken()
                        },
                        data: JSON.stringify(oAuthData),
                        success: function(data) {
                            resolve(data);
                        },
                        error: function(xhr) {
                            var errorMsg = this._securityUtils ? 
                                this._securityUtils.sanitizeErrorMessage(xhr.responseText) : 
                                "Authentication configuration failed";
                            reject(new Error(errorMsg));
                        }.bind(this)
                    });
                }.bind(this));
            }.bind(this)).then(function(data) {
                this._oAuthDialog.setBusy(false);
                this._oAuthDialog.close();
                var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                var sMessage = oBundle.getText("msg.authConfigurationSaved") || "Authentication configuration saved successfully";
                MessageToast.show(sMessage);
                this._extensionAPI.refresh();
                this._auditLogger.log("AUTH_CONFIGURATION_SAVED", "Authentication configuration updated", {
                    integrationId: oData.integrationId,
                    authMethod: oAuthData.authMethod
                });
            }.bind(this)).catch(function(error) {
                this._oAuthDialog.setBusy(false);
                MessageBox.error("Failed to save authentication configuration: " + error.message);
                this._auditLogger.log("AUTH_CONFIGURATION_FAILED", "Failed to save authentication configuration", {
                    integrationId: oData.integrationId,
                    error: error.message
                });
            }.bind(this));
        },

        /**
         * @function _validateAuthConfiguration
         * @description Validates authentication configuration data.
         * @param {Object} oData - Authentication data to validate
         * @returns {boolean} True if validation passes
         * @private
         */
        _validateAuthConfiguration: function(oData) {
            if (!oData.authMethod) {
                MessageBox.error("Authentication method is required");
                return false;
            }
            
            switch (oData.authMethod) {
                case "API_KEY":
                    if (!oData.apiKey || oData.apiKey.trim().length === 0) {
                        MessageBox.error("API Key is required");
                        return false;
                    }
                    break;
                case "BEARER_TOKEN":
                    if (!oData.bearerToken || oData.bearerToken.trim().length === 0) {
                        MessageBox.error("Bearer Token is required");
                        return false;
                    }
                    break;
                case "BASIC_AUTH":
                    if (!oData.basicAuth.username || !oData.basicAuth.password) {
                        MessageBox.error("Username and password are required for Basic Authentication");
                        return false;
                    }
                    break;
                case "OAUTH2":
                    if (!oData.oauth2.clientId || !oData.oauth2.clientSecret || !oData.oauth2.tokenUrl) {
                        MessageBox.error("Client ID, Client Secret, and Token URL are required for OAuth 2.0");
                        return false;
                    }
                    // Validate token URL format
                    if (!/^https:\/\/[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+$/.test(oData.oauth2.tokenUrl)) {
                        MessageBox.error("Token URL must be a valid HTTPS URL");
                        return false;
                    }
                    break;
            }
            
            return true;
        },

        /**
         * @function _sanitizeAuthConfiguration
         * @description Sanitizes authentication configuration for secure transmission.
         * @param {Object} oData - Authentication data to sanitize
         * @returns {Object} Sanitized authentication data
         * @private
         */
        _sanitizeAuthConfiguration: function(oData) {
            var oSanitized = {
                authMethod: this._securityUtils ? this._securityUtils.sanitizeInput(oData.authMethod) : oData.authMethod
            };
            
            switch (oData.authMethod) {
                case "API_KEY":
                    oSanitized.apiKey = oData.apiKey.trim();
                    break;
                case "BEARER_TOKEN":
                    oSanitized.bearerToken = oData.bearerToken.trim();
                    break;
                case "BASIC_AUTH":
                    oSanitized.basicAuth = {
                        username: this._securityUtils ? this._securityUtils.sanitizeInput(oData.basicAuth.username) : oData.basicAuth.username.trim(),
                        password: oData.basicAuth.password // Password not sanitized to preserve special characters
                    };
                    break;
                case "OAUTH2":
                    oSanitized.oauth2 = {
                        clientId: this._securityUtils ? this._securityUtils.sanitizeInput(oData.oauth2.clientId) : oData.oauth2.clientId.trim(),
                        clientSecret: oData.oauth2.clientSecret.trim(),
                        tokenUrl: this._securityUtils ? this._securityUtils.sanitizeInput(oData.oauth2.tokenUrl) : oData.oauth2.tokenUrl.trim(),
                        scope: this._securityUtils ? this._securityUtils.sanitizeInput(oData.oauth2.scope || '') : (oData.oauth2.scope || '').trim()
                    };
                    break;
                case "CUSTOM":
                    oSanitized.customHeaders = (oData.customHeaders || []).map(function(header) {
                        return {
                            name: this._securityUtils ? this._securityUtils.sanitizeInput(header.name) : header.name,
                            value: header.value // Header values may contain special characters
                        };
                    }.bind(this));
                    break;
            }
            
            return oSanitized;
        },

        /**
         * @function onCancelAuthConfiguration
         * @description Cancels authentication configuration and closes dialog.
         * @public
         */
        onCancelAuthConfiguration: function() {
            if (this._oAuthDialog) {
                this._oAuthDialog.close();
            }
        },

        /**
         * @function onRefreshLogs
         * @description Refreshes the logs display with current settings.
         * @public
         */
        onRefreshLogs: function() {
            if (this._oLogsDialog) {
                var oModel = this._oLogsDialog.getModel("logs");
                var oData = oModel.getData();
                this._loadIntegrationLogs(oData.integrationId, this._oLogsDialog);
            }
        },

        /**
         * @function onCloseLogs
         * @description Closes the logs viewer dialog.
         * @public
         */
        onCloseLogs: function() {
            if (this._oLogsDialog) {
                this._oLogsDialog.close();
            }
        },

        onGenerateInferences: function() {
            if (!this._securityUtils.hasRole("ReasoningManager")) {
                MessageBox.error("Access denied: Reasoning Manager role required");
                this._securityUtils.auditLog("GENERATE_INFERENCES_ACCESS_DENIED", { action: "generate_inferences" });
                return;
            }
            
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = this._securityUtils.sanitizeInput(oContext.getProperty("ID"));
            
            this._extensionAPI.getView().setBusy(true);
            
            const ajaxConfig = this._securityUtils.createSecureAjaxConfig({
                url: "/a2a/agent9/v1/tasks/" + encodeURIComponent(sTaskId) + "/infer",
                type: "POST",
                success: function(data) {
                    this._extensionAPI.getView().setBusy(false);
                    this._showInferenceResults(data);
                    this._securityUtils.auditLog("INFERENCES_GENERATED", { 
                        taskId: sTaskId,
                        inferenceCount: data.inferences ? data.inferences.length : 0
                    });
                }.bind(this),
                error: function(xhr) {
                    this._extensionAPI.getView().setBusy(false);
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    MessageBox.error("Inference generation failed: " + errorMsg);
                    this._securityUtils.auditLog("INFERENCE_GENERATION_FAILED", { 
                        taskId: sTaskId,
                        error: errorMsg
                    });
                }.bind(this)
            });
            
            jQuery.ajax(ajaxConfig);
        },

        _showInferenceResults: function(inferenceData) {
            var oView = this.base.getView();
            
            if (!this._oInferenceResultsDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent9.ext.fragment.InferenceResults",
                    controller: this
                }).then(function(oDialog) {
                    this._oInferenceResultsDialog = oDialog;
                    oView.addDependent(this._oInferenceResultsDialog);
                    
                    var oModel = new JSONModel(inferenceData);
                    this._oInferenceResultsDialog.setModel(oModel, "inference");
                    this._oInferenceResultsDialog.open();
                }.bind(this));
            } else {
                var oModel = new JSONModel(inferenceData);
                this._oInferenceResultsDialog.setModel(oModel, "inference");
                this._oInferenceResultsDialog.open();
            }
        },

        onMakeDecision: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            
            if (!this._oDecisionDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent9.ext.fragment.MakeDecision",
                    controller: this
                }).then(function(oDialog) {
                    this._oDecisionDialog = oDialog;
                    this.base.getView().addDependent(this._oDecisionDialog);
                    
                    var oModel = new JSONModel({
                        taskId: sTaskId,
                        decisionCriteria: [],
                        weightingStrategy: "EQUAL",
                        riskTolerance: "MEDIUM",
                        timeHorizon: "SHORT_TERM",
                        stakeholderPriorities: []
                    });
                    this._oDecisionDialog.setModel(oModel, "decision");
                    this._oDecisionDialog.open();
                    
                    this._loadDecisionOptions(sTaskId);
                }.bind(this));
            } else {
                this._oDecisionDialog.open();
                this._loadDecisionOptions(sTaskId);
            }
        },

        _loadDecisionOptions: function(sTaskId) {
            jQuery.ajax({
                url: "/a2a/agent9/v1/tasks/" + sTaskId + "/decision-options",
                type: "GET",
                success: function(data) {
                    var oModel = this._oDecisionDialog.getModel("decision");
                    var oData = oModel.getData();
                    oData.availableAlternatives = data.alternatives;
                    oData.availableCriteria = data.criteria;
                    oData.stakeholders = data.stakeholders;
                    oModel.setData(oData);
                }.bind(this),
                error: function(xhr) {
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    MessageBox.error("Failed to load decision options: " + errorMsg);
                }.bind(this)
            });
        },

        onValidateConclusion: function() {
            if (!this._securityUtils.hasRole("ReasoningValidator")) {
                MessageBox.error("Access denied: Reasoning Validator role required");
                this._securityUtils.auditLog("VALIDATE_CONCLUSION_ACCESS_DENIED", { action: "validate_conclusion" });
                return;
            }

            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            
            if (!sTaskId || typeof sTaskId !== "string" || sTaskId.trim() === "") {
                MessageBox.error("Invalid task ID");
                return;
            }
            
            this._extensionAPI.getView().setBusy(true);
            
            const ajaxConfig = this._securityUtils.createSecureAjaxConfig({
                url: "/a2a/agent9/v1/tasks/" + encodeURIComponent(sTaskId) + "/validate",
                type: "POST",
                success: function(data) {
                    this._extensionAPI.getView().setBusy(false);
                    this._showValidationResults(data);
                    this._securityUtils.auditLog("VALIDATE_CONCLUSION_SUCCESS", { taskId: sTaskId });
                }.bind(this),
                error: function(xhr) {
                    this._extensionAPI.getView().setBusy(false);
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    MessageBox.error("Validation failed: " + errorMsg);
                    this._securityUtils.auditLog("VALIDATE_CONCLUSION_FAILED", { taskId: sTaskId, error: xhr.status });
                }.bind(this)
            });
            
            jQuery.ajax(ajaxConfig);
        },

        _showValidationResults: function(validationData) {
            var sMessage = "Conclusion Validation Results:\\n\\n";
            
            sMessage += "Validation Status: " + this._securityUtils.encodeHTML(validationData.status || 'Unknown') + "\\n";
            sMessage += "Confidence Score: " + (parseFloat(validationData.confidence) || 0) + "%\\n";
            sMessage += "Logical Consistency: " + this._securityUtils.encodeHTML(validationData.consistency || 'Unknown') + "\\n";
            sMessage += "Supporting Evidence: " + (parseInt(validationData.supportingEvidence) || 0) + " facts\\n\\n";
            
            if (validationData.issues && validationData.issues.length > 0) {
                sMessage += "Validation Issues:\\n";
                validationData.issues.forEach(function(issue) {
                    const safeType = this._securityUtils.encodeHTML(issue.type || 'Unknown');
                    const safeDesc = this._securityUtils.encodeHTML(issue.description || 'No description');
                    sMessage += "• " + safeType + ": " + safeDesc + "\\n";
                }.bind(this));
            }
            
            if (validationData.recommendations && validationData.recommendations.length > 0) {
                sMessage += "\\nRecommendations:\\n";
                validationData.recommendations.forEach(function(rec) {
                    const safeRec = this._securityUtils.encodeHTML(rec || '');
                    sMessage += "• " + safeRec + "\\n";
                }.bind(this));
            }
            
            MessageBox.information(sMessage);
        },

        onExplainReasoning: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            
            if (!this._oExplanationDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent9.ext.fragment.ReasoningExplanation",
                    controller: this
                }).then(function(oDialog) {
                    this._oExplanationDialog = oDialog;
                    this.base.getView().addDependent(this._oExplanationDialog);
                    this._oExplanationDialog.open();
                    this._loadExplanationData(sTaskId);
                }.bind(this));
            } else {
                this._oExplanationDialog.open();
                this._loadExplanationData(sTaskId);
            }
        },

        _loadExplanationData: function(sTaskId) {
            jQuery.ajax({
                url: "/a2a/agent9/v1/tasks/" + sTaskId + "/explain",
                type: "GET",
                success: function(data) {
                    var oModel = new JSONModel({
                        explanation: data.explanation,
                        reasoningChain: data.reasoningChain,
                        factJustification: data.factJustification,
                        ruleApplications: data.ruleApplications,
                        confidenceBreakdown: data.confidenceBreakdown
                    });
                    this._oExplanationDialog.setModel(oModel, "explanation");
                    this._createExplanationVisualizations(data);
                }.bind(this),
                error: function(xhr) {
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    MessageBox.error("Failed to load explanation: " + errorMsg);
                }.bind(this)
            });
        },

        _createExplanationVisualizations: function(data) {
            // Create reasoning chain diagram
            this._createReasoningChainDiagram(data.reasoningChain);
            // Create confidence breakdown chart
            this._createConfidenceChart(data.confidenceBreakdown);
        },

        onOptimizeEngine: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            var sReasoningEngine = oContext.getProperty("reasoningEngine");
            
            MessageBox.confirm(
                "Optimize reasoning engine '" + sReasoningEngine + "' for this task?",
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._optimizeEngine(sTaskId);
                        }
                    }.bind(this)
                }
            );
        },

        _optimizeEngine: function(sTaskId) {
            if (!this._securityUtils.hasRole("ReasoningAdmin")) {
                MessageBox.error("Access denied: Reasoning Administrator role required");
                this._securityUtils.auditLog("OPTIMIZE_ENGINE_ACCESS_DENIED", { taskId: sTaskId });
                return;
            }

            if (!sTaskId || typeof sTaskId !== "string" || sTaskId.trim() === "") {
                MessageBox.error("Invalid task ID");
                return;
            }

            const ajaxConfig = this._securityUtils.createSecureAjaxConfig({
                url: "/a2a/agent9/v1/tasks/" + encodeURIComponent(sTaskId) + "/optimize",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    optimizationStrategy: "PERFORMANCE",
                    targetMetric: "INFERENCE_SPEED",
                    constraintHandling: "SOFT"
                }),
                success: function(data) {
                    const safePerformance = parseFloat(data.performanceImprovement) || 0;
                    const safeMemory = parseFloat(data.memoryReduction) || 0;
                    const safeAccuracy = parseFloat(data.accuracyMaintained) || 0;
                    MessageBox.success(
                        "Engine optimization completed!\\n" +
                        "Performance improvement: " + safePerformance + "%\\n" +
                        "Memory reduction: " + safeMemory + "%\\n" +
                        "Accuracy maintained: " + safeAccuracy + "%"
                    );
                    this._extensionAPI.refresh();
                    this._securityUtils.auditLog("OPTIMIZE_ENGINE_SUCCESS", { taskId: sTaskId, performance: safePerformance });
                }.bind(this),
                error: function(xhr) {
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    MessageBox.error("Engine optimization failed: " + errorMsg);
                    this._securityUtils.auditLog("OPTIMIZE_ENGINE_FAILED", { taskId: sTaskId, error: xhr.status });
                }.bind(this)
            });
            
            jQuery.ajax(ajaxConfig);
        },

        onUpdateKnowledge: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            
            if (!this._oKnowledgeUpdateDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent9.ext.fragment.UpdateKnowledge",
                    controller: this
                }).then(function(oDialog) {
                    this._oKnowledgeUpdateDialog = oDialog;
                    this.base.getView().addDependent(this._oKnowledgeUpdateDialog);
                    
                    var oModel = new JSONModel({
                        taskId: sTaskId,
                        updateType: "INCREMENTAL",
                        knowledgeSource: "EXTERNAL",
                        validationLevel: "STRICT",
                        conflictResolution: "MANUAL"
                    });
                    this._oKnowledgeUpdateDialog.setModel(oModel, "update");
                    this._oKnowledgeUpdateDialog.open();
                }.bind(this));
            } else {
                this._oKnowledgeUpdateDialog.open();
            }
        },

        onAnalyzeContradictions: function() {
            if (!this._securityUtils.hasRole("ReasoningAnalyst")) {
                MessageBox.error("Access denied: Reasoning Analyst role required");
                this._securityUtils.auditLog("ANALYZE_CONTRADICTIONS_ACCESS_DENIED", { action: "analyze_contradictions" });
                return;
            }

            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            
            if (!sTaskId || typeof sTaskId !== "string" || sTaskId.trim() === "") {
                MessageBox.error("Invalid task ID");
                return;
            }
            
            this._extensionAPI.getView().setBusy(true);
            
            const ajaxConfig = this._securityUtils.createSecureAjaxConfig({
                url: "/a2a/agent9/v1/tasks/" + encodeURIComponent(sTaskId) + "/analyze-contradictions",
                type: "POST",
                success: function(data) {
                    this._extensionAPI.getView().setBusy(false);
                    this._showContradictionAnalysis(data);
                    this._securityUtils.auditLog("ANALYZE_CONTRADICTIONS_SUCCESS", { taskId: sTaskId });
                }.bind(this),
                error: function(xhr) {
                    this._extensionAPI.getView().setBusy(false);
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    MessageBox.error("Contradiction analysis failed: " + errorMsg);
                    this._securityUtils.auditLog("ANALYZE_CONTRADICTIONS_FAILED", { taskId: sTaskId, error: xhr.status });
                }.bind(this)
            });
            
            jQuery.ajax(ajaxConfig);
        },

        _showContradictionAnalysis: function(analysisData) {
            var sMessage = "Contradiction Analysis Results:\\n\\n";
            
            if (analysisData.contradictions.length === 0) {
                sMessage += "No contradictions found in the knowledge base.\\n";
                sMessage += "Logical consistency: " + (parseFloat(analysisData.consistencyScore) || 0) + "%";
            } else {
                sMessage += "Found " + analysisData.contradictions.length + " contradictions:\\n\\n";
                
                analysisData.contradictions.slice(0, 5).forEach(function(contradiction, index) {
                    const safeDesc = this._securityUtils.encodeHTML(contradiction.description || 'Unknown contradiction');
                    const safeFacts = contradiction.facts ? 
                        contradiction.facts.map(f => this._securityUtils.encodeHTML(f)).join(", ") : 
                        'No facts';
                    const safeSeverity = this._securityUtils.encodeHTML(contradiction.severity || 'Unknown');
                    
                    sMessage += (index + 1) + ". " + safeDesc + "\\n";
                    sMessage += "   Conflicting facts: " + safeFacts + "\\n";
                    sMessage += "   Severity: " + safeSeverity + "\\n\\n";
                }.bind(this));
                
                if (analysisData.contradictions.length > 5) {
                    sMessage += "... and " + (analysisData.contradictions.length - 5) + " more contradictions\\n\\n";
                }
                
                sMessage += "Resolution strategies:\\n";
                analysisData.resolutionStrategies.forEach(function(strategy) {
                    const safeStrategy = this._securityUtils.encodeHTML(strategy || '');
                    sMessage += "• " + safeStrategy + "\\n";
                }.bind(this));
            }
            
            MessageBox.information(sMessage, {
                actions: ["Resolve Contradictions", MessageBox.Action.CLOSE],
                onClose: function(oAction) {
                    if (oAction === "Resolve Contradictions") {
                        this._resolveContradictions(analysisData.contradictions);
                    }
                }.bind(this)
            });
        },

        _resolveContradictions: function(contradictions) {
            if (!this._securityUtils.hasRole("ReasoningManager")) {
                MessageBox.error("Access denied: Reasoning Manager role required");
                this._securityUtils.auditLog("RESOLVE_CONTRADICTIONS_ACCESS_DENIED", { action: "resolve_contradictions" });
                return;
            }

            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            
            if (!sTaskId || typeof sTaskId !== "string" || sTaskId.trim() === "") {
                MessageBox.error("Invalid task ID");
                return;
            }
            
            const ajaxConfig = this._securityUtils.createSecureAjaxConfig({
                url: "/a2a/agent9/v1/tasks/" + encodeURIComponent(sTaskId) + "/resolve-contradictions",
                type: "POST",
                data: JSON.stringify({
                    contradictions: contradictions,
                    resolutionStrategy: "CONFIDENCE_BASED",
                    preserveConsistency: true
                }),
                success: function(data) {
                    const safeResolved = parseInt(data.resolvedCount) || 0;
                    const safeRemaining = parseInt(data.remainingCount) || 0;
                    const safeConsistency = parseFloat(data.newConsistencyScore) || 0;
                    MessageBox.success(
                        "Contradictions resolved successfully!\\n" +
                        "Resolved: " + safeResolved + "\\n" +
                        "Remaining: " + safeRemaining + "\\n" +
                        "Consistency improved to: " + safeConsistency + "%"
                    );
                    this._extensionAPI.refresh();
                    this._securityUtils.auditLog("RESOLVE_CONTRADICTIONS_SUCCESS", { taskId: sTaskId, resolved: safeResolved, remaining: safeRemaining });
                }.bind(this),
                error: function(xhr) {
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    MessageBox.error("Failed to resolve contradictions: " + errorMsg);
                    this._securityUtils.auditLog("RESOLVE_CONTRADICTIONS_FAILED", { taskId: sTaskId, error: xhr.status });
                }.bind(this)
            });
            
            jQuery.ajax(ajaxConfig);
        },

        onConfirmDecision: function() {
            if (!this._securityUtils.hasRole("DecisionMaker")) {
                MessageBox.error("Access denied: Decision Maker role required");
                this._securityUtils.auditLog("CONFIRM_DECISION_ACCESS_DENIED", { action: "confirm_decision" });
                return;
            }

            var oModel = this._oDecisionDialog.getModel("decision");
            var oData = oModel.getData();
            
            if (!oData.decisionCriteria || oData.decisionCriteria.length === 0) {
                MessageBox.error("Please define decision criteria");
                return;
            }

            if (!oData.taskId || typeof oData.taskId !== "string" || oData.taskId.trim() === "") {
                MessageBox.error("Invalid task ID");
                return;
            }
            
            this._oDecisionDialog.setBusy(true);
            
            const ajaxConfig = this._securityUtils.createSecureAjaxConfig({
                url: "/a2a/agent9/v1/tasks/" + encodeURIComponent(oData.taskId) + "/decide",
                type: "POST",
                data: JSON.stringify(oData),
                success: function(data) {
                    this._oDecisionDialog.setBusy(false);
                    this._oDecisionDialog.close();
                    
                    const safeAction = this._securityUtils.encodeHTML(data.recommendedAction || 'Unknown');
                    const safeConfidence = parseFloat(data.confidence) || 0;
                    const safeOutcome = this._securityUtils.encodeHTML(data.expectedOutcome || 'Unknown');
                    MessageBox.success(
                        "Decision made successfully!\\n" +
                        "Recommended action: " + safeAction + "\\n" +
                        "Confidence: " + safeConfidence + "%\\n" +
                        "Expected outcome: " + safeOutcome
                    );
                    
                    this._extensionAPI.refresh();
                    this._securityUtils.auditLog("CONFIRM_DECISION_SUCCESS", { taskId: oData.taskId, action: safeAction, confidence: safeConfidence });
                }.bind(this),
                error: function(xhr) {
                    this._oDecisionDialog.setBusy(false);
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    MessageBox.error("Decision making failed: " + errorMsg);
                    this._securityUtils.auditLog("CONFIRM_DECISION_FAILED", { taskId: oData.taskId, error: xhr.status });
                }.bind(this)
            });
            
            jQuery.ajax(ajaxConfig);
        },

        onConfirmKnowledgeUpdate: function() {
            if (!this._securityUtils.hasRole("KnowledgeManager")) {
                MessageBox.error("Access denied: Knowledge Manager role required");
                this._securityUtils.auditLog("CONFIRM_KNOWLEDGE_UPDATE_ACCESS_DENIED", { action: "confirm_knowledge_update" });
                return;
            }

            var oModel = this._oKnowledgeUpdateDialog.getModel("update");
            var oData = oModel.getData();
            
            if (!oData.taskId || typeof oData.taskId !== "string" || oData.taskId.trim() === "") {
                MessageBox.error("Invalid task ID");
                return;
            }
            
            this._oKnowledgeUpdateDialog.setBusy(true);
            
            const ajaxConfig = this._securityUtils.createSecureAjaxConfig({
                url: "/a2a/agent9/v1/tasks/" + encodeURIComponent(oData.taskId) + "/update-knowledge",
                type: "POST",
                data: JSON.stringify(oData),
                success: function(data) {
                    this._oKnowledgeUpdateDialog.setBusy(false);
                    this._oKnowledgeUpdateDialog.close();
                    
                    const safeFacts = parseInt(data.factsAdded) || 0;
                    const safeRules = parseInt(data.rulesUpdated) || 0;
                    const safeScore = parseFloat(data.consistencyScore) || 0;
                    MessageBox.success(
                        "Knowledge base updated successfully!\\n" +
                        "New facts added: " + safeFacts + "\\n" +
                        "Rules updated: " + safeRules + "\\n" +
                        "Consistency score: " + safeScore + "%"
                    );
                    
                    this._extensionAPI.refresh();
                    this._securityUtils.auditLog("CONFIRM_KNOWLEDGE_UPDATE_SUCCESS", { taskId: oData.taskId, factsAdded: safeFacts, rulesUpdated: safeRules });
                }.bind(this),
                error: function(xhr) {
                    this._oKnowledgeUpdateDialog.setBusy(false);
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    MessageBox.error("Knowledge update failed: " + errorMsg);
                    this._securityUtils.auditLog("CONFIRM_KNOWLEDGE_UPDATE_FAILED", { taskId: oData.taskId, error: xhr.status });
                }.bind(this)
            });
            
            jQuery.ajax(ajaxConfig);
        },

        // Placeholder visualization functions for future chart implementations
        _createTestResultsChart: function(results) {
            // Placeholder for connection test results visualization
        },

        _createExecutionTimeline: function(timeline) {
            // Placeholder for integration execution timeline
        },

        _createPerformanceMetrics: function(metrics) {
            // Placeholder for performance metrics visualization
        },

        _createErrorAnalysisChart: function(errors) {
            // Placeholder for error analysis visualization
        }
    });
});