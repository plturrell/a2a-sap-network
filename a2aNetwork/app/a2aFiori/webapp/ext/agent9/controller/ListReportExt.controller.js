sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/ui/core/Fragment",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/model/json/JSONModel",
    "sap/base/Log",
    "a2a/network/agent9/ext/utils/SecurityUtils"
], (ControllerExtension, Fragment, MessageBox, MessageToast, JSONModel, Log, SecurityUtils) => {
    "use strict";

    /**
     * @class a2a.network.agent9.ext.controller.ListReportExt
     * @extends sap.ui.core.mvc.ControllerExtension
     * @description Controller extension for Agent 9 List Report - Advanced Reasoning and Decision Making.
     * Provides sophisticated AI reasoning capabilities including knowledge management, inference engines,
     * logical analysis, decision making, and problem solving with enterprise-grade security and accessibility.
     */
    return ControllerExtension.extend("a2a.network.agent9.ext.controller.ListReportExt", {

        override: {
            /**
             * @function onInit
             * @description Initializes the controller extension with security utilities, device model,
             * dialog caching, and real-time updates.
             * @override
             */
            onInit() {
                this._extensionAPI = this.base.getExtensionAPI();
                this._securityUtils = SecurityUtils;
                this._initializeDeviceModel();
                this._initializeDialogCache();
                this._initializePerformanceOptimizations();
                this._startRealtimeUpdates();
            },

            /**
             * @function onExit
             * @description Cleanup resources on controller destruction.
             * @override
             */
            onExit() {
                this._cleanupResources();
                if (this.base.onExit) {
                    this.base.onExit.apply(this, arguments);
                }
            }
        },

        // Dialog caching for performance
        _dialogCache: {},

        // Error recovery configuration
        _errorRecoveryConfig: {
            maxRetries: 3,
            retryDelay: 1000,
            exponentialBackoff: true
        },

        /**
         * @function _initializeDeviceModel
         * @description Sets up device model for responsive design.
         * @private
         */
        _initializeDeviceModel() {
            const oDeviceModel = new sap.ui.model.json.JSONModel(sap.ui.Device);
            this.base.getView().setModel(oDeviceModel, "device");
        },

        /**
         * @function _initializeDialogCache
         * @description Initializes dialog cache for performance.
         * @private
         */
        _initializeDialogCache() {
            this._dialogCache = {};
        },

        /**
         * @function _initializePerformanceOptimizations
         * @description Sets up performance optimization features.
         * @private
         */
        _initializePerformanceOptimizations() {
            // Throttle dashboard updates
            this._throttledDashboardUpdate = this._throttle(this._loadDashboardData.bind(this), 1000);
            // Debounce search operations
            this._debouncedSearch = this._debounce(this._performSearch.bind(this), 300);
        },

        /**
         * @function _throttle
         * @description Creates a throttled function.
         * @param {Function} fn - Function to throttle
         * @param {number} limit - Time limit in milliseconds
         * @returns {Function} Throttled function
         * @private
         */
        _throttle(fn, limit) {
            let inThrottle;
            return function() {
                const args = arguments;
                const context = this;
                if (!inThrottle) {
                    fn.apply(context, args);
                    inThrottle = true;
                    setTimeout(() => { inThrottle = false; }, limit);
                }
            };
        },

        /**
         * @function _debounce
         * @description Creates a debounced function.
         * @param {Function} fn - Function to debounce
         * @param {number} delay - Delay in milliseconds
         * @returns {Function} Debounced function
         * @private
         */
        _debounce(fn, delay) {
            let timeoutId;
            return function() {
                const context = this;
                const args = arguments;
                clearTimeout(timeoutId);
                timeoutId = setTimeout(() => {
                    fn.apply(context, args);
                }, delay);
            };
        },

        /**
         * @function _performSearch
         * @description Performs search operation (placeholder for search functionality).
         * @param {string} sQuery - Search query
         * @private
         */
        _performSearch(sQuery) {
            // Implement search logic for reasoning tasks and knowledge base
        },

        /**
         * @function onCreateIntegration
         * @description Opens dialog to create new API integration task.
         * @public
         */
        onCreateIntegration() {
            // Security check: Validate user authorization for creating integrations
            if (!this._hasRole("IntegrationAdmin")) {
                MessageBox.error("Access denied: Insufficient privileges for creating API integrations");
                this._auditLog("CREATE_INTEGRATION_ACCESS_DENIED", "User attempted integration creation without IntegrationAdmin role");
                return;
            }

            this._getOrCreateDialog("createIntegration", "a2a.network.agent9.ext.fragment.CreateIntegration")
                .then((oDialog) => {
                    const oModel = new JSONModel({
                        integrationName: "",
                        description: "",
                        integrationType: "REST_API",
                        endpointUrl: "",
                        authMethod: "API_KEY",
                        protocol: "HTTPS",
                        timeout: 30000,
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
                        priority: "MEDIUM",
                        // Validation states
                        integrationNameState: "None",
                        endpointUrlState: "None",
                        integrationTypeState: "None",
                        integrationNameStateText: "",
                        endpointUrlStateText: "",
                        integrationTypeStateText: ""
                    });
                    oDialog.setModel(oModel, "create");
                    oDialog.open();
                    this._loadIntegrationOptions(oDialog);

                    this._auditLog("CREATE_INTEGRATION_INITIATED", "Integration creation dialog opened");
                })
                .catch((error) => {
                    MessageBox.error(`Failed to open integration creation dialog: ${ error.message}`);
                    this._auditLog("CREATE_INTEGRATION_ERROR", "Failed to open dialog", { error: error.message });
                });
        },

        /**
         * @function onManageEndpoints
         * @description Opens endpoint management interface for API configurations.
         * @public
         */
        onManageEndpoints() {
            // Security check: Validate user authorization for managing endpoints
            if (!this._hasRole("IntegrationAdmin")) {
                MessageBox.error("Access denied: Insufficient privileges for managing endpoints");
                this._auditLog("MANAGE_ENDPOINTS_ACCESS_DENIED", "User attempted endpoint management without IntegrationAdmin role");
                return;
            }

            this._getOrCreateDialog("manageEndpoints", "a2a.network.agent9.ext.fragment.EndpointManager")
                .then((oDialog) => {
                    oDialog.open();
                    this._loadEndpointData(oDialog);

                    this._auditLog("MANAGE_ENDPOINTS_OPENED", "Endpoint management dialog opened");
                })
                .catch((error) => {
                    MessageBox.error(`Failed to open endpoint management: ${ error.message}`);
                    this._auditLog("MANAGE_ENDPOINTS_ERROR", "Failed to open dialog", { error: error.message });
                });
        },

        /**
         * @function onTestConnections
         * @description Tests API connections for selected integrations.
         * @public
         */
        onTestConnections() {
            const oTable = this._extensionAPI.getTable();
            const aSelectedContexts = oTable.getSelectedContexts();

            if (aSelectedContexts.length === 0) {
                const oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                const sMessage = oBundle.getText("msg.selectIntegrationsForTest") || "Please select at least one integration to test.";
                MessageBox.warning(sMessage);
                return;
            }

            // Security check: Validate user authorization for testing connections
            if (!this._hasRole("IntegrationUser")) {
                MessageBox.error("Access denied: Insufficient privileges for testing connections");
                this._auditLog("TEST_CONNECTIONS_ACCESS_DENIED", "User attempted connection test without IntegrationUser role");
                return;
            }

            // Limit bulk operations for security
            if (aSelectedContexts.length > 20) {
                MessageBox.error("Connection testing limited to 20 integrations maximum for security reasons");
                this._auditLog("TEST_CONNECTIONS_LIMIT_EXCEEDED", "Connection test attempted on too many integrations", {
                    selectedCount: aSelectedContexts.length,
                    maxAllowed: 20
                });
                return;
            }

            const aIntegrationIds = aSelectedContexts.map((oContext) => {
                const sIntegrationId = oContext.getProperty("integrationId");
                return this._securityUtils ? this._securityUtils.sanitizeInput(sIntegrationId) : sIntegrationId;
            });

            // Validate integration IDs for security
            const invalidIds = aIntegrationIds.filter((id) => {
                return !this._securityUtils.validateIntegrationId(id);
            });

            if (invalidIds.length > 0) {
                MessageBox.error(`Invalid integration IDs detected: ${ invalidIds.join(", ")}`);
                this._auditLog("TEST_CONNECTIONS_INVALID_IDS", "Invalid integration IDs in bulk test", { invalidIds });
                return;
            }

            this._getOrCreateDialog("testConnections", "a2a.network.agent9.ext.fragment.ConnectionTester")
                .then((oDialog) => {
                    const oModel = new JSONModel({
                        selectedIntegrations: aIntegrationIds,
                        testResults: [],
                        testInProgress: false,
                        testParameters: {
                            timeout: 30000,
                            includeAuth: true,
                            validateSsl: true,
                            checkResponseStructure: true
                        }
                    });
                    oDialog.setModel(oModel, "test");
                    oDialog.open();
                    this._startConnectionTests(aIntegrationIds, oDialog);

                    this._auditLog("TEST_CONNECTIONS_STARTED", "Connection testing initiated", {
                        integrationCount: aIntegrationIds.length
                    });
                })
                .catch((error) => {
                    MessageBox.error(`Failed to start connection tests: ${ error.message}`);
                    this._auditLog("TEST_CONNECTIONS_ERROR", "Failed to start tests", { error: error.message });
                });
        },

        /**
         * @function _getOrCreateDialog
         * @description Gets cached dialog or creates new one with accessibility and responsive features.
         * @param {string} sDialogId - Dialog identifier
         * @param {string} sFragmentName - Fragment name
         * @returns {Promise<sap.m.Dialog>} Promise resolving to dialog
         * @private
         */
        _getOrCreateDialog(sDialogId, sFragmentName) {
            const that = this;

            if (this._dialogCache[sDialogId]) {
                return Promise.resolve(this._dialogCache[sDialogId]);
            }

            return Fragment.load({
                id: this.base.getView().getId(),
                name: sFragmentName,
                controller: this
            }).then((oDialog) => {
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
        _enableDialogAccessibility(oDialog) {
            oDialog.addEventDelegate({
                onAfterRendering() {
                    const $dialog = oDialog.$();

                    // Set tabindex for focusable elements
                    $dialog.find("input, button, select, textarea").attr("tabindex", "0");

                    // Handle escape key
                    $dialog.on("keydown", (e) => {
                        if (e.key === "Escape") {
                            oDialog.close();
                        }
                    });

                    // Focus first input on open
                    setTimeout(() => {
                        $dialog.find("input:visible:first").focus();
                    }, 100);
                }
            });
        },

        /**
         * @function _optimizeDialogForDevice
         * @description Optimizes dialog for current device.
         * @param {sap.m.Dialog} oDialog - Dialog to optimize
         * @private
         */
        _optimizeDialogForDevice(oDialog) {
            if (sap.ui.Device.system.phone) {
                oDialog.setStretch(true);
                oDialog.setContentWidth("100%");
                oDialog.setContentHeight("100%");
            } else if (sap.ui.Device.system.tablet) {
                oDialog.setContentWidth("95%");
                oDialog.setContentHeight("90%");
            }
        },

        /**
         * @function _withErrorRecovery
         * @description Wraps operation with error recovery.
         * @param {Function} fnOperation - Operation to execute
         * @param {Object} oOptions - Recovery options
         * @returns {Promise} Promise with error recovery
         * @private
         */
        _withErrorRecovery(fnOperation, oOptions) {
            const that = this;
            const oConfig = Object.assign({}, this._errorRecoveryConfig, oOptions);

            function attempt(retriesLeft, delay) {
                return fnOperation().catch((error) => {
                    if (retriesLeft > 0) {
                        const oBundle = that.base.getView().getModel("i18n").getResourceBundle();
                        const sRetryMsg = oBundle.getText("recovery.retrying") || "Network error. Retrying...";
                        MessageToast.show(sRetryMsg);

                        return new Promise((resolve) => {
                            setTimeout(resolve, delay);
                        }).then(() => {
                            const nextDelay = oConfig.exponentialBackoff ? delay * 2 : delay;
                            return attempt(retriesLeft - 1, nextDelay);
                        });
                    }
                    throw error;
                });
            }

            return attempt(oConfig.maxRetries, oConfig.retryDelay);
        },

        /**
         * @function _cleanupResources
         * @description Cleans up resources to prevent memory leaks.
         * @private
         */
        _cleanupResources() {
            // Clean up event sources
            if (this._realtimeEventSource) {
                this._realtimeEventSource.close();
                this._realtimeEventSource = null;
            }
            if (this._testEventSource) {
                this._testEventSource.close();
                this._testEventSource = null;
            }

            // Clean up cached dialogs
            Object.keys(this._dialogCache).forEach((key) => {
                if (this._dialogCache[key]) {
                    this._dialogCache[key].destroy();
                }
            });
            this._dialogCache = {};

            this._auditLog("RESOURCES_CLEANED_UP", "Controller resources cleaned up on exit");
        },

        /**
         * @function _loadIntegrationOptions
         * @description Loads API integration options and configuration templates.
         * @param {sap.m.Dialog} oDialog - Target dialog (optional)
         * @private
         */
        _loadIntegrationOptions(oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["createIntegration"];
            if (!oTargetDialog) {return;}

            this._withErrorRecovery(() => {
                return new Promise((resolve, reject) => {
                    // Create secure AJAX configuration
                    const ajaxConfig = {
                        url: "/a2a/agent9/v1/integration-options",
                        type: "GET",
                        headers: {
                            "X-Requested-With": "XMLHttpRequest",
                            "Accept": "application/json"
                        },
                        timeout: 15000,
                        success: function(data) {
                            // Sanitize response data
                            const sanitizedData = {
                                integrationTypes: this._sanitizeArray(data.integrationTypes || []),
                                authMethods: this._sanitizeArray(data.authMethods || []),
                                protocols: this._sanitizeArray(data.protocols || []),
                                dataFormats: this._sanitizeArray(data.dataFormats || []),
                                templates: this._sanitizeArray(data.templates || [])
                            };

                            const oModel = oTargetDialog.getModel("create");
                            const oData = oModel.getData();
                            Object.assign(oData, sanitizedData);
                            oModel.setData(oData);
                            resolve(sanitizedData);
                        }.bind(this),
                        error: function(xhr, textStatus, errorThrown) {
                            let errorMsg = "Network error";
                            if (xhr.responseText) {
                                errorMsg = this._securityUtils ?
                                    this._securityUtils.sanitizeErrorMessage(xhr.responseText) :
                                    "Server error";
                            }
                            reject(new Error(`Failed to load integration options: ${ errorMsg}`));
                        }.bind(this)
                    };

                    jQuery.ajax(ajaxConfig);
                });
            }).catch((error) => {
                MessageBox.error(error.message);
                this._auditLog("LOAD_INTEGRATION_OPTIONS_FAILED", "Failed to load options", { error: error.message });
            });
        },

        /**
         * @function _loadEndpointData
         * @description Loads endpoint management data including configurations and health status.
         * @param {sap.m.Dialog} oDialog - Target dialog (optional)
         * @private
         */
        _loadEndpointData(oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["manageEndpoints"];
            if (!oTargetDialog) {return;}

            oTargetDialog.setBusy(true);

            this._withErrorRecovery(() => {
                return new Promise((resolve, reject) => {
                    jQuery.ajax({
                        url: "/a2a/agent9/v1/endpoints",
                        type: "GET",
                        headers: {
                            "X-Requested-With": "XMLHttpRequest",
                            "Accept": "application/json"
                        },
                        timeout: 15000,
                        success: function(data) {
                            const oModel = new JSONModel({
                                endpoints: this._sanitizeArray(data.endpoints || []),
                                healthStatus: this._sanitizeObject(data.healthStatus || {}),
                                configurations: this._sanitizeArray(data.configurations || []),
                                apiVersions: this._sanitizeArray(data.apiVersions || []),
                                rateLimits: this._sanitizeObject(data.rateLimits || {})
                            });
                            oTargetDialog.setModel(oModel, "endpoints");
                            resolve(data);
                        }.bind(this),
                        error: function(xhr) {
                            const errorMsg = this._securityUtils ?
                                this._securityUtils.sanitizeErrorMessage(xhr.responseText) :
                                "Failed to load endpoint data";
                            reject(new Error(errorMsg));
                        }.bind(this)
                    });
                });
            }).then((data) => {
                oTargetDialog.setBusy(false);
                this._createEndpointVisualizations(data, oTargetDialog);
                this._auditLog("ENDPOINT_DATA_LOADED", "Endpoint data loaded successfully", { endpointCount: data.endpoints.length });
            }).catch((error) => {
                oTargetDialog.setBusy(false);
                MessageBox.error(error.message);
                this._auditLog("ENDPOINT_DATA_LOAD_FAILED", "Failed to load endpoint data", { error: error.message });
            });
        },

        /**
         * @function _startConnectionTests
         * @description Starts connection testing for selected integrations.
         * @param {Array} aIntegrationIds - Array of integration IDs to test
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _startConnectionTests(aIntegrationIds, oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["testConnections"];
            if (!oTargetDialog || !aIntegrationIds.length) {return;}

            const oModel = oTargetDialog.getModel("test");
            const oData = oModel.getData();
            oData.testInProgress = true;
            oData.testResults = [];
            oModel.setData(oData);

            // Start real-time monitoring of connection tests
            this._startTestMonitoring(aIntegrationIds, oTargetDialog);

            this._withErrorRecovery(() => {
                return new Promise((resolve, reject) => {
                    jQuery.ajax({
                        url: "/a2a/agent9/v1/test-connections",
                        type: "POST",
                        contentType: "application/json",
                        headers: {
                            "X-Requested-With": "XMLHttpRequest",
                            "X-CSRF-Token": this._getCSRFToken()
                        },
                        data: JSON.stringify({
                            integrationIds: aIntegrationIds,
                            testParameters: oData.testParameters
                        }),
                        timeout: 120000, // 2 minutes for bulk testing
                        success: function(data) {
                            oData.testInProgress = false;
                            oData.testResults = this._sanitizeArray(data.results || []);
                            oModel.setData(oData);
                            resolve(data);
                        }.bind(this),
                        error: function(xhr) {
                            oData.testInProgress = false;
                            oModel.setData(oData);
                            const errorMsg = this._securityUtils ?
                                this._securityUtils.sanitizeErrorMessage(xhr.responseText) :
                                "Connection test failed";
                            reject(new Error(errorMsg));
                        }.bind(this)
                    });
                });
            }).then((data) => {
                this._processTestResults(data.results, oTargetDialog);
                this._auditLog("CONNECTION_TESTS_COMPLETED", "Connection tests completed", {
                    integrationCount: aIntegrationIds.length,
                    successCount: data.results.filter(r => r.success).length
                });
            }).catch((error) => {
                const oModel = oTargetDialog.getModel("test");
                const oData = oModel.getData();
                oData.testInProgress = false;
                oModel.setData(oData);
                MessageBox.error(`Connection test failed: ${ error.message}`);
                this._auditLog("CONNECTION_TESTS_FAILED", "Connection tests failed", { error: error.message });
            });
        },

        /**
         * @function _startTestMonitoring
         * @description Starts real-time monitoring for connection tests.
         * @param {Array} aIntegrationIds - Array of integration IDs being tested
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _startTestMonitoring(aIntegrationIds, oDialog) {
            if (this._testEventSource) {
                this._testEventSource.close();
            }

            // Validate EventSource URL for security
            const testStreamUrl = `/a2a/agent9/v1/test-stream?ids=${ encodeURIComponent(aIntegrationIds.join(","))}`;
            if (!this._securityUtils || !this._securityUtils.validateEventSourceUrl(testStreamUrl)) {
                return;
            }

            this._testEventSource = new EventSource(testStreamUrl);

            this._testEventSource.onmessage = function(event) {
                try {
                    const data = JSON.parse(event.data);
                    this._updateTestProgress(data, oDialog);
                } catch (error) {
                    // Error handled in test progress
                }
            }.bind(this);

            this._testEventSource.onerror = function() {
                const oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                const sMessage = oBundle.getText("error.testMonitoringLost") || "Test monitoring connection lost";
                MessageToast.show(sMessage);
            }.bind(this);
        },

        /**
         * @function _updateTestProgress
         * @description Updates test progress display with real-time data.
         * @param {Object} testData - Real-time test progress data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _updateTestProgress(testData, oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["testConnections"];
            if (!oTargetDialog) {return;}

            const oModel = oTargetDialog.getModel("test");
            if (!oModel) {return;}

            const oData = oModel.getData();

            if (testData.type === "progress_update") {
                const integrationIndex = oData.testResults.findIndex((result) => {
                    return result.integrationId === testData.integrationId;
                });

                if (integrationIndex >= 0) {
                    oData.testResults[integrationIndex] = this._sanitizeObject(testData.result);
                } else {
                    oData.testResults.push(this._sanitizeObject(testData.result));
                }
                oModel.setData(oData);
            }
        },

        /**
         * @function _processTestResults
         * @description Processes and displays connection test results.
         * @param {Array} aResults - Array of test results
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _processTestResults(aResults, oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["testConnections"];
            if (!oTargetDialog || !aResults) {return;}

            const successCount = aResults.filter((result) => {
                return result.success;
            }).length;

            const oBundle = this.base.getView().getModel("i18n").getResourceBundle();
            let sMessage = oBundle.getText("msg.connectionTestResults") || "Connection tests completed";
            sMessage += `: ${ successCount }/${ aResults.length } successful`;
            MessageToast.show(sMessage);

            // Create test results visualization
            this._createTestResultsChart(aResults, oTargetDialog);
        },

        /**
         * @function _createTestResultsChart
         * @description Creates visualization chart for connection test results.
         * @param {Array} aResults - Array of test results
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createTestResultsChart(aResults, oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["testConnections"];
            if (!oTargetDialog) {return;}

            const oChart = oTargetDialog.byId("testResultsChart");
            if (!oChart || !aResults) {return;}

            const aChartData = aResults.map((result) => {
                return {
                    Integration: result.integrationName || result.integrationId,
                    Status: result.success ? "Success" : "Failed",
                    ResponseTime: result.responseTimeMs || 0,
                    ErrorCode: result.errorCode || ""
                };
            });

            const oChartModel = new sap.ui.model.json.JSONModel({
                testData: aChartData
            });
            oChart.setModel(oChartModel);
        },

        /**
         * @function _createEndpointVisualizations
         * @description Creates endpoint health and performance visualizations.
         * @param {Object} data - Endpoint data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createEndpointVisualizations(data, oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["manageEndpoints"];
            if (!oTargetDialog) {return;}

            this._createEndpointHealthChart(data.healthStatus, oTargetDialog);
            this._createRateLimitChart(data.rateLimits, oTargetDialog);
        },

        /**
         * @function _createEndpointHealthChart
         * @description Creates endpoint health status visualization.
         * @param {Object} healthData - Health status data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createEndpointHealthChart(healthData, oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["manageEndpoints"];
            if (!oTargetDialog) {return;}

            const oHealthChart = oTargetDialog.byId("endpointHealthChart");
            if (!oHealthChart || !healthData) {return;}

            const aChartData = Object.keys(healthData).map((endpoint) => {
                const health = healthData[endpoint];
                return {
                    Endpoint: endpoint,
                    Uptime: health.uptimePercent || 0,
                    ResponseTime: health.avgResponseTime || 0,
                    ErrorRate: health.errorRate || 0
                };
            });

            const oChartModel = new sap.ui.model.json.JSONModel({
                healthData: aChartData
            });
            oHealthChart.setModel(oChartModel);

            const oBundle = this.base.getView().getModel("i18n").getResourceBundle();
            oHealthChart.setVizProperties({
                categoryAxis: {
                    title: { text: oBundle.getText("chart.endpoints") || "Endpoints" }
                },
                valueAxis: {
                    title: { text: oBundle.getText("chart.uptime") || "Uptime %" }
                },
                title: {
                    text: oBundle.getText("chart.endpointHealth") || "Endpoint Health Status"
                }
            });
        },

        /**
         * @function _loadDashboardData
         * @description Loads reasoning dashboard data with AI metrics and performance analytics.
         * @param {sap.m.Dialog} oDialog - Target dialog (optional)
         * @private
         */
        _loadDashboardData(oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["reasoningDashboard"];
            if (!oTargetDialog) {return;}

            oTargetDialog.setBusy(true);

            this._withErrorRecovery(() => {
                return new Promise((resolve, reject) => {
                    jQuery.ajax({
                        url: "/a2a/agent9/v1/dashboard",
                        type: "GET",
                        success(data) {
                            const oDashboardModel = new JSONModel({
                                summary: data.summary,
                                reasoningMetrics: data.reasoningMetrics,
                                knowledgeBase: data.knowledgeBase,
                                enginePerformance: data.enginePerformance,
                                inferenceTrends: data.inferenceTrends,
                                decisionAccuracy: data.decisionAccuracy
                            });

                            oTargetDialog.setModel(oDashboardModel, "dashboard");
                            resolve(data);
                        },
                        error(xhr) {
                            const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                            reject(new Error(`Failed to load dashboard data: ${ errorMsg}`));
                        }
                    });
                });
            }).then((data) => {
                oTargetDialog.setBusy(false);
                this._createDashboardCharts(data, oTargetDialog);
            }).catch((error) => {
                oTargetDialog.setBusy(false);
                MessageBox.error(error.message);
            });
        },

        /**
         * @function _createDashboardCharts
         * @description Creates reasoning analytics visualization charts.
         * @param {Object} data - Dashboard data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createDashboardCharts(data, oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["reasoningDashboard"];
            if (!oTargetDialog) {return;}

            this._createConfidenceTrendsChart(data.inferenceTrends, oTargetDialog);
            this._createEnginePerformanceChart(data.enginePerformance, oTargetDialog);
            this._createDecisionAccuracyChart(data.decisionAccuracy, oTargetDialog);
        },

        /**
         * @function _createConfidenceTrendsChart
         * @description Creates confidence trends visualization chart.
         * @param {Object} trendsData - Inference trends data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createConfidenceTrendsChart(trendsData, oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["reasoningDashboard"];
            if (!oTargetDialog) {return;}

            const oVizFrame = oTargetDialog.byId("confidenceTrendsChart");
            if (!oVizFrame || !trendsData) {return;}

            const aChartData = trendsData.map((trend) => {
                return {
                    Time: trend.timestamp,
                    Confidence: trend.averageConfidence,
                    Inferences: trend.inferencesGenerated,
                    Accuracy: trend.accuracy
                };
            });

            const oChartModel = new sap.ui.model.json.JSONModel({
                trendsData: aChartData
            });
            oVizFrame.setModel(oChartModel);

            const oBundle = this.base.getView().getModel("i18n").getResourceBundle();
            oVizFrame.setVizProperties({
                categoryAxis: {
                    title: { text: oBundle.getText("chart.time") || "Time" }
                },
                valueAxis: {
                    title: { text: oBundle.getText("chart.confidencePercent") || "Confidence %" }
                },
                title: {
                    text: oBundle.getText("chart.confidenceTrends") || "Reasoning Confidence Trends"
                }
            });
        },

        /**
         * @function onKnowledgeManager
         * @description Opens knowledge management interface for facts, rules, and ontologies.
         * @public
         */
        onKnowledgeManager() {
            this._getOrCreateDialog("knowledgeManager", "a2a.network.agent9.ext.fragment.KnowledgeManager")
                .then((oDialog) => {
                    oDialog.open();
                    this._loadKnowledgeData(oDialog);
                })
                .catch((error) => {
                    MessageBox.error(`Failed to open Knowledge Manager: ${ error.message}`);
                });
        },

        /**
         * @function _loadKnowledgeData
         * @description Loads knowledge base data including facts, rules, and ontologies.
         * @param {sap.m.Dialog} oDialog - Target dialog (optional)
         * @private
         */
        _loadKnowledgeData(oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["knowledgeManager"];
            if (!oTargetDialog) {return;}

            oTargetDialog.setBusy(true);

            this._withErrorRecovery(() => {
                return new Promise((resolve, reject) => {
                    jQuery.ajax({
                        url: "/a2a/agent9/v1/knowledge-base",
                        type: "GET",
                        success(data) {
                            const oModel = new JSONModel({
                                facts: data.facts,
                                rules: data.rules,
                                ontologies: data.ontologies,
                                consistency: data.consistency,
                                completeness: data.completeness,
                                domains: data.domains
                            });
                            oTargetDialog.setModel(oModel, "knowledge");
                            resolve(data);
                        },
                        error(xhr) {
                            const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                            reject(new Error(`Failed to load knowledge data: ${ errorMsg}`));
                        }
                    });
                });
            }).then((data) => {
                oTargetDialog.setBusy(false);
                this._createKnowledgeVisualizations(data, oTargetDialog);
            }).catch((error) => {
                oTargetDialog.setBusy(false);
                MessageBox.error(error.message);
            });
        },

        /**
         * @function _createKnowledgeVisualizations
         * @description Creates knowledge base growth and distribution visualizations.
         * @param {Object} data - Knowledge data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createKnowledgeVisualizations(data, oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["knowledgeManager"];
            if (!oTargetDialog) {return;}

            const oKnowledgeChart = oTargetDialog.byId("knowledgeGrowthChart");
            if (!oKnowledgeChart || !data.growth) {return;}

            const aChartData = data.growth.map((point) => {
                return {
                    Date: point.date,
                    Facts: point.factsCount,
                    Rules: point.rulesCount,
                    Inferences: point.inferencesCount
                };
            });

            const oChartModel = new sap.ui.model.json.JSONModel({
                growthData: aChartData
            });
            oKnowledgeChart.setModel(oChartModel);

            const oBundle = this.base.getView().getModel("i18n").getResourceBundle();
            oKnowledgeChart.setVizProperties({
                categoryAxis: {
                    title: { text: oBundle.getText("chart.time") || "Time" }
                },
                valueAxis: {
                    title: { text: oBundle.getText("chart.count") || "Count" }
                },
                title: {
                    text: oBundle.getText("chart.knowledgeGrowth") || "Knowledge Base Growth"
                }
            });
        },

        /**
         * @function onRuleEngine
         * @description Opens rule engine configuration interface for managing inference rules.
         * @public
         */
        onRuleEngine() {
            this._getOrCreateDialog("ruleEngine", "a2a.network.agent9.ext.fragment.RuleEngine")
                .then((oDialog) => {
                    oDialog.open();
                    this._loadRuleData(oDialog);
                })
                .catch((error) => {
                    MessageBox.error(`Failed to open Rule Engine: ${ error.message}`);
                });
        },

        /**
         * @function _loadRuleData
         * @description Loads rule engine data including rules and performance metrics.
         * @param {sap.m.Dialog} oDialog - Target dialog (optional)
         * @private
         */
        _loadRuleData(oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["ruleEngine"];
            if (!oTargetDialog) {return;}

            oTargetDialog.setBusy(true);

            this._withErrorRecovery(() => {
                return new Promise((resolve, reject) => {
                    jQuery.ajax({
                        url: "/a2a/agent9/v1/rules",
                        type: "GET",
                        success(data) {
                            const oModel = new JSONModel({
                                rules: data.rules,
                                ruleTypes: data.ruleTypes,
                                conflictResolution: data.conflictResolution,
                                rulePerformance: data.performance
                            });
                            oTargetDialog.setModel(oModel, "rules");
                            resolve(data);
                        },
                        error(xhr) {
                            const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                            reject(new Error(`Failed to load rule data: ${ errorMsg}`));
                        }
                    });
                });
            }).then((data) => {
                oTargetDialog.setBusy(false);
                this._createRuleVisualizations(data, oTargetDialog);
            }).catch((error) => {
                oTargetDialog.setBusy(false);
                MessageBox.error(error.message);
            });
        },

        /**
         * @function onInferenceEngine
         * @description Opens inference engine interface for managing logical inferences.
         * @public
         */
        onInferenceEngine() {
            this._getOrCreateDialog("inferenceEngine", "a2a.network.agent9.ext.fragment.InferenceEngine")
                .then((oDialog) => {
                    oDialog.open();
                    this._loadInferenceData(oDialog);
                })
                .catch((error) => {
                    MessageBox.error(`Failed to open Inference Engine: ${ error.message}`);
                });
        },

        /**
         * @function _loadInferenceData
         * @description Loads inference engine data including inference chains and validation.
         * @param {sap.m.Dialog} oDialog - Target dialog (optional)
         * @private
         */
        _loadInferenceData(oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["inferenceEngine"];
            if (!oTargetDialog) {return;}

            oTargetDialog.setBusy(true);

            this._withErrorRecovery(() => {
                return new Promise((resolve, reject) => {
                    jQuery.ajax({
                        url: "/a2a/agent9/v1/inferences",
                        type: "GET",
                        success(data) {
                            const oModel = new JSONModel({
                                inferences: data.inferences,
                                inferenceChains: data.chains,
                                confidence: data.confidence,
                                validation: data.validation
                            });
                            oTargetDialog.setModel(oModel, "inference");
                            resolve(data);
                        },
                        error(xhr) {
                            const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                            reject(new Error(`Failed to load inference data: ${ errorMsg}`));
                        }
                    });
                });
            }).then((data) => {
                oTargetDialog.setBusy(false);
                this._createInferenceVisualizations(data, oTargetDialog);
            }).catch((error) => {
                oTargetDialog.setBusy(false);
                MessageBox.error(error.message);
            });
        },

        /**
         * @function _createInferenceVisualizations
         * @description Creates inference chain and confidence distribution visualizations.
         * @param {Object} data - Inference data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createInferenceVisualizations(data, oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["inferenceEngine"];
            if (!oTargetDialog) {return;}

            this._createInferenceChainDiagram(data.chains, oTargetDialog);
            this._createConfidenceDistribution(data.confidence, oTargetDialog);
        },

        /**
         * @function _createInferenceChainDiagram
         * @description Creates network diagram showing inference relationships.
         * @param {Array} chains - Inference chains data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createInferenceChainDiagram(chains, oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["inferenceEngine"];
            if (!oTargetDialog || !chains) {return;}

            const oChainChart = oTargetDialog.byId("inferenceChainChart");
            if (!oChainChart) {return;}

            const aChartData = chains.map((chain) => {
                return {
                    ChainId: chain.id,
                    Steps: chain.steps.length,
                    Confidence: chain.confidence,
                    Depth: chain.depth
                };
            });

            const oChainModel = new sap.ui.model.json.JSONModel({
                chainData: aChartData
            });
            oChainChart.setModel(oChainModel);
        },

        /**
         * @function onDecisionMaker
         * @description Opens decision-making interface for multi-criteria analysis.
         * @public
         */
        onDecisionMaker() {
            this._getOrCreateDialog("decisionMaker", "a2a.network.agent9.ext.fragment.DecisionMaker")
                .then((oDialog) => {
                    oDialog.open();
                    this._loadDecisionData(oDialog);
                })
                .catch((error) => {
                    MessageBox.error(`Failed to open Decision Maker: ${ error.message}`);
                });
        },

        /**
         * @function _loadDecisionData
         * @description Loads decision-making data including criteria and alternatives.
         * @param {sap.m.Dialog} oDialog - Target dialog (optional)
         * @private
         */
        _loadDecisionData(oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["decisionMaker"];
            if (!oTargetDialog) {return;}

            oTargetDialog.setBusy(true);

            this._withErrorRecovery(() => {
                return new Promise((resolve, reject) => {
                    jQuery.ajax({
                        url: "/a2a/agent9/v1/decisions",
                        type: "GET",
                        success(data) {
                            const oModel = new JSONModel({
                                decisions: data.decisions,
                                criteria: data.criteria,
                                alternatives: data.alternatives,
                                recommendations: data.recommendations
                            });
                            oTargetDialog.setModel(oModel, "decision");
                            resolve(data);
                        },
                        error(xhr) {
                            const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                            reject(new Error(`Failed to load decision data: ${ errorMsg}`));
                        }
                    });
                });
            }).then((data) => {
                oTargetDialog.setBusy(false);
                this._createDecisionVisualizations(data, oTargetDialog);
            }).catch((error) => {
                oTargetDialog.setBusy(false);
                MessageBox.error(error.message);
            });
        },

        /**
         * @function onProblemSolver
         * @description Opens problem-solving interface for complex reasoning scenarios.
         * @public
         */
        onProblemSolver() {
            this._getOrCreateDialog("problemSolver", "a2a.network.agent9.ext.fragment.ProblemSolver")
                .then((oDialog) => {
                    oDialog.open();
                    this._loadProblemData(oDialog);
                })
                .catch((error) => {
                    MessageBox.error(`Failed to open Problem Solver: ${ error.message}`);
                });
        },

        /**
         * @function _loadProblemData
         * @description Loads problem-solving data including strategies and complexity metrics.
         * @param {sap.m.Dialog} oDialog - Target dialog (optional)
         * @private
         */
        _loadProblemData(oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["problemSolver"];
            if (!oTargetDialog) {return;}

            oTargetDialog.setBusy(true);

            this._withErrorRecovery(() => {
                return new Promise((resolve, reject) => {
                    jQuery.ajax({
                        url: "/a2a/agent9/v1/problems",
                        type: "GET",
                        success(data) {
                            const oModel = new JSONModel({
                                problems: data.problems,
                                solutions: data.solutions,
                                strategies: data.strategies,
                                complexity: data.complexity
                            });
                            oTargetDialog.setModel(oModel, "problem");
                            resolve(data);
                        },
                        error(xhr) {
                            const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                            reject(new Error(`Failed to load problem data: ${ errorMsg}`));
                        }
                    });
                });
            }).then((data) => {
                oTargetDialog.setBusy(false);
                this._createProblemVisualizations(data, oTargetDialog);
            }).catch((error) => {
                oTargetDialog.setBusy(false);
                MessageBox.error(error.message);
            });
        },

        /**
         * @function onLogicalAnalyzer
         * @description Opens logical analyzer for consistency checks and contradiction detection.
         * @public
         */
        onLogicalAnalyzer() {
            this._getOrCreateDialog("logicalAnalyzer", "a2a.network.agent9.ext.fragment.LogicalAnalyzer")
                .then((oDialog) => {
                    oDialog.open();
                    this._loadAnalysisData(oDialog);
                })
                .catch((error) => {
                    MessageBox.error(`Failed to open Logical Analyzer: ${ error.message}`);
                });
        },

        /**
         * @function _loadAnalysisData
         * @description Loads logical analysis data including contradictions and consistency checks.
         * @param {sap.m.Dialog} oDialog - Target dialog (optional)
         * @private
         */
        _loadAnalysisData(oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["logicalAnalyzer"];
            if (!oTargetDialog) {return;}

            oTargetDialog.setBusy(true);

            this._withErrorRecovery(() => {
                return new Promise((resolve, reject) => {
                    jQuery.ajax({
                        url: "/a2a/agent9/v1/analysis",
                        type: "GET",
                        success(data) {
                            const oModel = new JSONModel({
                                contradictions: data.contradictions,
                                consistencyChecks: data.consistencyChecks,
                                logicalErrors: data.logicalErrors,
                                optimization: data.optimization
                            });
                            oTargetDialog.setModel(oModel, "analysis");
                            resolve(data);
                        },
                        error(xhr) {
                            const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                            reject(new Error(`Failed to load analysis data: ${ errorMsg}`));
                        }
                    });
                });
            }).then((data) => {
                oTargetDialog.setBusy(false);
                this._createAnalysisVisualizations(data, oTargetDialog);
            }).catch((error) => {
                oTargetDialog.setBusy(false);
                MessageBox.error(error.message);
            });
        },

        /**
         * @function _startRealtimeUpdates
         * @description Starts real-time updates for API integration events and status changes.
         * @private
         */
        _startRealtimeUpdates() {
            // Validate EventSource URL for security
            const eventSourceUrl = "/a2a/agent9/v1/realtime-updates";
            if (this._securityUtils && !this._securityUtils.validateEventSourceUrl(eventSourceUrl)) {
                MessageBox.error("Invalid real-time update URL");
                this._auditLog("REALTIME_UPDATES_BLOCKED", "Invalid EventSource URL blocked", { url: eventSourceUrl });
                return;
            }

            this._realtimeEventSource = new EventSource(eventSourceUrl);

            this._realtimeEventSource.onmessage = function(event) {
                try {
                    const data = JSON.parse(event.data);

                    if (data.type === "integration_complete") {
                        const safeIntegrationName = this._securityUtils ?
                            this._securityUtils.encodeHTML(data.integrationName || "Unknown integration") :
                            (data.integrationName || "Unknown integration");
                        const oResourceBundle = this.base.getView().getModel("i18n").getResourceBundle();
                        const sMessage = oResourceBundle.getText("msg.integrationCompleted") || "Integration completed";
                        MessageToast.show(`${sMessage }: ${ safeIntegrationName}`);
                        this._extensionAPI.refresh();
                    } else if (data.type === "connection_test_result") {
                        const safeEndpointName = this._securityUtils ?
                            this._securityUtils.encodeHTML(data.endpointName || "Unknown endpoint") :
                            (data.endpointName || "Unknown endpoint");
                        const oResourceBundle2 = this.base.getView().getModel("i18n").getResourceBundle();
                        const sMessage2 = data.success ?
                            (oResourceBundle2.getText("msg.connectionTestSuccess") || "Connection test successful") :
                            (oResourceBundle2.getText("msg.connectionTestFailed") || "Connection test failed");
                        MessageToast.show(`${sMessage2 }: ${ safeEndpointName}`);
                    } else if (data.type === "rate_limit_exceeded") {
                        const oResourceBundle3 = this.base.getView().getModel("i18n").getResourceBundle();
                        const sMessage3 = oResourceBundle3.getText("msg.rateLimitExceeded") || "Rate limit exceeded for API endpoint";
                        MessageToast.show(sMessage3);
                    } else if (data.type === "authentication_failure") {
                        const oResourceBundle4 = this.base.getView().getModel("i18n").getResourceBundle();
                        const sMessage4 = oResourceBundle4.getText("msg.authenticationFailure") || "Authentication failure detected";
                        MessageToast.show(sMessage4);
                        this._auditLog("AUTHENTICATION_FAILURE_DETECTED", "Real-time authentication failure event", data);
                    }
                } catch (error) {
                    // // console.error("Error processing real-time update:", error);
                    this._auditLog("REALTIME_UPDATE_ERROR", "Error processing real-time update", { error: error.message });
                }
            }.bind(this);

            this._realtimeEventSource.onerror = function() {
                const oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                const sMessage = oBundle.getText("msg.realtimeDisconnected") || "Real-time updates disconnected";
                MessageToast.show(sMessage);
                this._auditLog("REALTIME_UPDATES_DISCONNECTED", "EventSource connection lost");
            }.bind(this);

            this._auditLog("REALTIME_UPDATES_STARTED", "EventSource connection established");
        },

        /**
         * @function onConfirmCreateIntegration
         * @description Confirms and creates a new API integration with validation.
         * @public
         */
        onConfirmCreateIntegration() {
            const oDialog = this._dialogCache["createIntegration"];
            if (!oDialog) {
                MessageBox.error("Create integration dialog not found");
                return;
            }

            const oModel = oDialog.getModel("create");
            const oData = oModel.getData();

            // Validate required fields
            const validation = this._validateCreateIntegrationData(oData);
            if (!validation.isValid) {
                MessageBox.error(validation.message);
                return;
            }

            // Validate integration parameters with SecurityUtils
            if (this._securityUtils && this._securityUtils.validateIntegrationParameters) {
                const paramValidation = this._securityUtils.validateIntegrationParameters(oData);
                if (!paramValidation.isValid) {
                    MessageBox.error(`Invalid integration parameters: ${ paramValidation.message}`);
                    return;
                }
            }

            // Sanitize input data
            const sanitizedData = this._sanitizeCreateIntegrationData(oData);

            oDialog.setBusy(true);

            this._withErrorRecovery(() => {
                return new Promise((resolve, reject) => {
                    jQuery.ajax({
                        url: "/a2a/agent9/v1/integrations",
                        type: "POST",
                        contentType: "application/json",
                        headers: {
                            "X-Requested-With": "XMLHttpRequest",
                            "X-CSRF-Token": this._getCSRFToken()
                        },
                        data: JSON.stringify(sanitizedData),
                        timeout: 30000,
                        success(data) {
                            resolve(data);
                        },
                        error: function(xhr, textStatus, errorThrown) {
                            let errorMsg = "Network error";
                            if (xhr.responseText) {
                                errorMsg = this._securityUtils ?
                                    this._securityUtils.sanitizeErrorMessage(xhr.responseText) :
                                    "Server error";
                            }
                            reject(new Error(errorMsg));
                        }.bind(this)
                    });
                });
            }).then((data) => {
                oDialog.setBusy(false);
                oDialog.close();
                const oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                const sMessage = oBundle.getText("msg.integrationCreated") || "API integration created successfully";
                MessageToast.show(sMessage);
                this._extensionAPI.refresh();
                this._auditLog("INTEGRATION_CREATED", "API integration created successfully", {
                    integrationName: sanitizedData.integrationName,
                    integrationType: sanitizedData.integrationType
                });
            }).catch((error) => {
                oDialog.setBusy(false);
                const oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                const sErrorMsg = oBundle.getText("error.createIntegrationFailed") || "Failed to create integration";
                MessageBox.error(`${sErrorMsg }: ${ error.message}`);
                this._auditLog("INTEGRATION_CREATE_FAILED", "Failed to create API integration", { error: error.message });
            });
        },

        /**
         * @function onCancelCreateIntegration
         * @description Cancels integration creation and closes dialog.
         * @public
         */
        onCancelCreateIntegration() {
            const oDialog = this._dialogCache["createIntegration"];
            if (oDialog) {
                oDialog.close();
            }
        },

        /**
         * @function _validateCreateIntegrationData
         * @description Validates integration creation data.
         * @param {Object} oData - Integration data to validate
         * @returns {Object} Validation result with isValid flag and message
         * @private
         */
        _validateCreateIntegrationData(oData) {
            const oResourceBundle = this.base.getView().getModel("i18n").getResourceBundle();

            if (!oData.integrationName || !oData.integrationName.trim()) {
                const sMessage = oResourceBundle.getText("validation.integrationNameRequired") || "Integration name is required";
                return { isValid: false, message: sMessage };
            }

            // Validate integration name format
            if (!/^[a-zA-Z0-9\s\-_]{3,50}$/.test(oData.integrationName)) {
                const sMessage = oResourceBundle.getText("validation.integrationNameFormat") || "Integration name must be 3-50 characters (alphanumeric, spaces, hyphens, underscores)";
                return { isValid: false, message: sMessage };
            }

            if (!oData.integrationType || !oData.integrationType.trim()) {
                const sMessage = oResourceBundle.getText("validation.integrationTypeRequired") || "Integration type is required";
                return { isValid: false, message: sMessage };
            }

            if (!oData.endpointUrl || !oData.endpointUrl.trim()) {
                const sMessage = oResourceBundle.getText("validation.endpointUrlRequired") || "Endpoint URL is required";
                return { isValid: false, message: sMessage };
            }

            // Validate URL format and security
            if (!/^https:\/\/[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+$/.test(oData.endpointUrl)) {
                const sMessage = oResourceBundle.getText("validation.endpointUrlFormat") || "Endpoint URL must be a valid HTTPS URL";
                return { isValid: false, message: sMessage };
            }

            return { isValid: true };
        },

        /**
         * @function _sanitizeCreateIntegrationData
         * @description Sanitizes integration creation data for security.
         * @param {Object} oData - Integration data to sanitize
         * @returns {Object} Sanitized integration data
         * @private
         */
        _sanitizeCreateIntegrationData(oData) {
            return {
                integrationName: this._securityUtils ?
                    this._securityUtils.sanitizeInput(oData.integrationName) : oData.integrationName.trim(),
                description: this._securityUtils ?
                    this._securityUtils.sanitizeInput(oData.description || "") : (oData.description || "").trim(),
                integrationType: this._securityUtils ?
                    this._securityUtils.sanitizeInput(oData.integrationType) : oData.integrationType,
                endpointUrl: this._securityUtils ?
                    this._securityUtils.sanitizeInput(oData.endpointUrl) : oData.endpointUrl.trim(),
                authMethod: this._securityUtils ?
                    this._securityUtils.sanitizeInput(oData.authMethod) : oData.authMethod,
                protocol: this._securityUtils ?
                    this._securityUtils.sanitizeInput(oData.protocol || "HTTPS") : (oData.protocol || "HTTPS"),
                timeout: Math.max(1000, Math.min(300000, parseInt(oData.timeout, 10) || 30000)),
                retryPolicy: {
                    maxRetries: Math.max(0, Math.min(10, parseInt(oData.retryPolicy?.maxRetries, 10) || 3)),
                    retryDelay: Math.max(100, Math.min(30000, parseInt(oData.retryPolicy?.retryDelay, 10) || 1000)),
                    exponentialBackoff: Boolean(oData.retryPolicy?.exponentialBackoff)
                },
                rateLimiting: {
                    requestsPerSecond: Math.max(1, Math.min(1000,
                        parseInt(oData.rateLimiting?.requestsPerSecond, 10) || 10)),
                    burstSize: Math.max(1, Math.min(10000, parseInt(oData.rateLimiting?.burstSize, 10) || 50)),
                    enabled: Boolean(oData.rateLimiting?.enabled)
                },
                dataFormat: this._securityUtils ?
                    this._securityUtils.sanitizeInput(oData.dataFormat || "JSON") : (oData.dataFormat || "JSON"),
                validation: {
                    validateSchema: Boolean(oData.validation?.validateSchema),
                    strictMode: Boolean(oData.validation?.strictMode),
                    sanitizeInput: Boolean(oData.validation?.sanitizeInput)
                },
                monitoring: {
                    logRequests: Boolean(oData.monitoring?.logRequests),
                    trackPerformance: Boolean(oData.monitoring?.trackPerformance),
                    alertOnFailure: Boolean(oData.monitoring?.alertOnFailure)
                },
                priority: this._securityUtils ? this._securityUtils.sanitizeInput(oData.priority || "MEDIUM") : (oData.priority || "MEDIUM")
            };
        },

        /**
         * @function _createRuleVisualizations
         * @description Creates rule engine performance and distribution visualizations.
         * @param {Object} data - Rule data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createRuleVisualizations(data, oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["ruleEngine"];
            if (!oTargetDialog || !data.performance) {return;}

            const oRuleChart = oTargetDialog.byId("rulePerformanceChart");
            if (!oRuleChart) {return;}

            const aChartData = data.performance.map((perf) => {
                return {
                    RuleType: perf.type,
                    ExecutionTime: perf.avgExecutionTime,
                    Accuracy: perf.accuracy,
                    Usage: perf.usageCount
                };
            });

            const oChartModel = new sap.ui.model.json.JSONModel({
                performanceData: aChartData
            });
            oRuleChart.setModel(oChartModel);
        },

        /**
         * @function _createDecisionVisualizations
         * @description Creates decision criteria and alternatives visualizations.
         * @param {Object} data - Decision data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createDecisionVisualizations(data, oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["decisionMaker"];
            if (!oTargetDialog || !data.criteria) {return;}

            const oDecisionChart = oTargetDialog.byId("decisionCriteriaChart");
            if (!oDecisionChart) {return;}

            const aChartData = data.criteria.map((criterion) => {
                return {
                    Criterion: criterion.name,
                    Weight: criterion.weight,
                    Impact: criterion.impact
                };
            });

            const oChartModel = new sap.ui.model.json.JSONModel({
                criteriaData: aChartData
            });
            oDecisionChart.setModel(oChartModel);
        },

        /**
         * @function _createProblemVisualizations
         * @description Creates problem complexity and solution visualizations.
         * @param {Object} data - Problem data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createProblemVisualizations(data, oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["problemSolver"];
            if (!oTargetDialog || !data.complexity) {return;}

            const oProblemChart = oTargetDialog.byId("problemComplexityChart");
            if (!oProblemChart) {return;}

            const aChartData = data.complexity.map((complex) => {
                return {
                    Domain: complex.domain,
                    Complexity: complex.score,
                    SolutionTime: complex.avgSolutionTime,
                    Success: complex.successRate
                };
            });

            const oChartModel = new sap.ui.model.json.JSONModel({
                complexityData: aChartData
            });
            oProblemChart.setModel(oChartModel);
        },

        /**
         * @function _createAnalysisVisualizations
         * @description Creates logical analysis and consistency visualizations.
         * @param {Object} data - Analysis data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createAnalysisVisualizations(data, oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["logicalAnalyzer"];
            if (!oTargetDialog || !data.consistencyChecks) {return;}

            const oAnalysisChart = oTargetDialog.byId("consistencyTrendsChart");
            if (!oAnalysisChart) {return;}

            const aChartData = data.consistencyChecks.map((check) => {
                return {
                    Timestamp: check.timestamp,
                    Consistency: check.score,
                    Contradictions: check.contradictionCount,
                    Resolved: check.resolvedCount
                };
            });

            const oChartModel = new sap.ui.model.json.JSONModel({
                consistencyData: aChartData
            });
            oAnalysisChart.setModel(oChartModel);
        },

        /**
         * @function _createConfidenceDistribution
         * @description Creates confidence level distribution visualization.
         * @param {Object} confidence - Confidence data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createConfidenceDistribution(confidence, oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["inferenceEngine"];
            if (!oTargetDialog || !confidence) {return;}

            const oConfidenceChart = oTargetDialog.byId("confidenceDistributionChart");
            if (!oConfidenceChart) {return;}

            const aChartData = confidence.distribution.map((dist) => {
                return {
                    Range: dist.range,
                    Count: dist.count,
                    Percentage: dist.percentage
                };
            });

            const oChartModel = new sap.ui.model.json.JSONModel({
                distributionData: aChartData
            });
            oConfidenceChart.setModel(oChartModel);
        },

        /**
         * @function _hasRole
         * @description Checks if current user has specified role.
         * @param {string} role - Role to check
         * @returns {boolean} True if user has role
         * @private
         */
        _hasRole(role) {
            const user = sap.ushell?.Container?.getUser();
            if (user && user.hasRole) {
                return user.hasRole(role);
            }
            // Mock role validation for development/testing
            const mockRoles = ["IntegrationAdmin", "IntegrationUser", "IntegrationOperator"];
            return mockRoles.includes(role);
        },

        /**
         * @function _auditLog
         * @description Logs security and operational events for audit purposes.
         * @param {string} action - Action being performed
         * @param {string} description - Description of the action
         * @param {Object} details - Additional details (optional)
         * @private
         */
        _auditLog(action, description, details) {
            const user = this._getCurrentUser();
            const timestamp = new Date().toISOString();
            const logEntry = {
                timestamp,
                user,
                agent: "Agent9_APIIntegration",
                action,
                description,
                details: details || {}
            };
            Log.info(`AUDIT: ${ JSON.stringify(logEntry)}`);
        },

        /**
         * @function _getCurrentUser
         * @description Gets current user ID for audit logging.
         * @returns {string} User ID or "anonymous"
         * @private
         */
        _getCurrentUser() {
            return sap.ushell?.Container?.getUser()?.getId() || "anonymous";
        },

        /**
         * @function _getCSRFToken
         * @description Gets CSRF token for secure POST requests.
         * @returns {string} CSRF token
         * @private
         */
        _getCSRFToken() {
            // In a real implementation, this would fetch the actual CSRF token
            // For now, return a placeholder that would be handled by the security utils
            return this._securityUtils ? this._securityUtils.getCSRFToken() : "placeholder-token";
        },

        /**
         * @function _sanitizeArray
         * @description Sanitizes array elements for security.
         * @param {Array} arr - Array to sanitize
         * @returns {Array} Sanitized array
         * @private
         */
        _sanitizeArray(arr) {
            if (!Array.isArray(arr)) {return [];}
            return arr.map((item) => {
                if (typeof item === "string") {
                    return this._securityUtils ? this._securityUtils.sanitizeInput(item) : item;
                } else if (typeof item === "object" && item !== null) {
                    return this._sanitizeObject(item);
                }
                return item;

            });
        },

        /**
         * @function _sanitizeObject
         * @description Recursively sanitizes object properties for security.
         * @param {Object} obj - Object to sanitize
         * @returns {Object} Sanitized object
         * @private
         */
        _sanitizeObject(obj) {
            if (!obj || typeof obj !== "object") {return {};}
            const sanitized = {};
            Object.keys(obj).forEach((key) => {
                if (typeof obj[key] === "string") {
                    sanitized[key] = this._securityUtils ? this._securityUtils.sanitizeInput(obj[key]) : obj[key];
                } else if (Array.isArray(obj[key])) {
                    sanitized[key] = this._sanitizeArray(obj[key]);
                } else if (typeof obj[key] === "object" && obj[key] !== null) {
                    sanitized[key] = this._sanitizeObject(obj[key]);
                } else {
                    sanitized[key] = obj[key];
                }
            });
            return sanitized;
        }
    });
});