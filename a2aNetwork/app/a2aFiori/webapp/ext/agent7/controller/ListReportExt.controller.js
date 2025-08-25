sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/ui/core/Fragment",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/model/json/JSONModel",
    "sap/base/security/encodeXML",
    "sap/base/security/encodeURL",
    "sap/base/Log",
    "../utils/SecurityUtils"
], (ControllerExtension, Fragment, MessageBox, MessageToast, JSONModel, encodeXML, encodeURL, Log, SecurityUtils) => {
    "use strict";

    /**
     * @class a2a.network.agent7.ext.controller.ListReportExt
     * @extends sap.ui.core.mvc.ControllerExtension
     * @description Controller extension for Agent 7 List Report - Agent Management and Orchestration.
     * Provides agent lifecycle management, health monitoring, performance analysis, and coordination features.
     */
    return ControllerExtension.extend("a2a.network.agent7.ext.controller.ListReportExt", {

        override: {
            /**
             * @function onInit
             * @description Initializes the controller with real-time updates and performance optimizations.
             * @override
             */
            onInit() {
                this._extensionAPI = this.base.getExtensionAPI();
                this._securityUtils = SecurityUtils;
                this._initializeDeviceModel();
                this._initializeDialogCache();
                this._initializePerformanceOptimizations();
                this._startRealtimeUpdates();

                // Initialize resource bundle for i18n
                this._oResourceBundle = this.base.getView().getModel("i18n").getResourceBundle();
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
            // Implement search logic
        },

        /**
         * @function _hasRole
         * @description Checks if user has required role for operation
         * @param {string} sRole - Required role
         * @returns {boolean} True if user has role
         * @private
         */
        _hasRole(sRole) {
            // In a real implementation, this would check actual user roles
            // For now, we'll check against a mock role system
            const aUserRoles = this._getUserRoles();
            return aUserRoles.includes(sRole);
        },

        /**
         * @function _getUserRoles
         * @description Gets current user's roles
         * @returns {Array<string>} Array of user roles
         * @private
         */
        _getUserRoles() {
            // This would typically come from the authentication service
            // Mock implementation for security testing
            return ["AgentUser", "AgentAdmin"]; // Mock admin role for now
        },

        /**
         * @function _authorizeOperation
         * @description Authorizes specific operations
         * @param {string} sOperation - Operation to authorize
         * @returns {boolean} True if authorized
         * @private
         */
        _authorizeOperation(sOperation) {
            const mRequiredRoles = {
                "BULK_OPERATIONS": ["AgentAdmin"],
                "UPDATE_AGENT": ["AgentAdmin", "AgentOperator"],
                "DELETE_AGENT": ["AgentAdmin"],
                "START_AGENT": ["AgentAdmin", "AgentOperator"],
                "STOP_AGENT": ["AgentAdmin", "AgentOperator"],
                "CONFIG_AGENT": ["AgentAdmin"],
                "VIEW_HEALTH": ["AgentAdmin", "AgentOperator", "AgentUser"],
                "COORDINATION": ["AgentAdmin"]
            };

            const aRequiredRoles = mRequiredRoles[sOperation] || [];
            const aUserRoles = this._getUserRoles();

            return aRequiredRoles.some((sRole) => {
                return aUserRoles.includes(sRole);
            });
        },

        /**
         * @function _logSecurityEvent
         * @description Logs security events for audit
         * @param {string} sEvent - Event type
         * @param {string} sDescription - Description
         * @param {Object} oData - Additional data
         * @private
         */
        _logSecurityEvent(sEvent, sDescription, oData) {
            if (this._securityUtils && this._securityUtils.logSecurityEvent) {
                this._securityUtils.logSecurityEvent(sEvent, sDescription, oData);
            }
        },

        /**
         * @function _getOrCreateDialog
         * @description Gets cached dialog or creates new one.
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
                        const sRetryMsg = oBundle.getText("recovery.retrying");
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
            if (this._healthEventSource) {
                this._healthEventSource.close();
                this._healthEventSource = null;
            }

            // Clean up cached dialogs
            Object.keys(this._dialogCache).forEach((key) => {
                if (this._dialogCache[key]) {
                    this._dialogCache[key].destroy();
                }
            });
            this._dialogCache = {};

            // Clean up legacy dialog references
            const aDialogs = ["_oCreateDialog", "_oDashboard", "_oRegisterDialog",
                "_oHealthMonitor", "_oPerformanceDialog", "_oCoordinatorDialog",
                "_oBulkDialog"];
            aDialogs.forEach((sDialog) => {
                if (this[sDialog]) {
                    this[sDialog].destroy();
                    this[sDialog] = null;
                }
            });
        },

        /**
         * @function onCreateManagementTask
         * @description Opens dialog to create new agent management task.
         * @public
         */
        onCreateManagementTask() {
            this._getOrCreateDialog("createManagementTask", "a2a.network.agent7.ext.fragment.CreateManagementTask")
                .then((oDialog) => {
                    const oModel = new JSONModel({
                        taskName: "",
                        description: "",
                        managedAgent: "",
                        operationType: "",
                        priority: "MEDIUM",
                        scheduledTime: null,
                        parameters: {},
                        notifyOnCompletion: true
                    });
                    oDialog.setModel(oModel, "create");
                    oDialog.open();
                    this._loadAvailableAgents(oDialog);
                });
        },

        /**
         * @function _loadAvailableAgents
         * @description Loads available agents for selection.
         * @private
         */
        _loadAvailableAgents(oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["createManagementTask"];
            if (!oTargetDialog) {return;}

            this._withErrorRecovery(() => {
                return new Promise((resolve, reject) => {
                    // SECURITY FIX: Use secure AJAX request instead of direct jQuery.ajax
                    if (this._securityUtils) {
                        this._securityUtils.secureAjaxRequest({
                            url: "/a2a/agent7/v1/registered-agents",
                            type: "GET"
                        }).then((data) => {
                            const oModel = oTargetDialog.getModel("create");
                            const oData = oModel.getData();
                            // Sanitize received data
                            oData.availableAgents = data.agents ? data.agents.map((agent) => {
                                return {
                                    id: this._securityUtils.sanitizeInput(agent.id),
                                    name: this._securityUtils.sanitizeInput(agent.name),
                                    type: this._securityUtils.sanitizeInput(agent.type)
                                };
                            }) : [];
                            oData.availableOperations = data.operations ? data.operations.map((op) => {
                                return this._securityUtils.sanitizeInput(op);
                            }) : [];
                            oModel.setData(oData);
                            resolve(data);
                        }).catch((xhr) => {
                            reject(new Error(`Failed to load available agents: ${ xhr.responseText || "Network error"}`));
                        });
                    } else {
                        // Fallback to regular AJAX with basic headers
                        jQuery.ajax({
                            url: "/a2a/agent7/v1/registered-agents",
                            type: "GET",
                            headers: {
                                "X-Requested-With": "XMLHttpRequest"
                            },
                            success(data) {
                                const oModel = oTargetDialog.getModel("create");
                                const oData = oModel.getData();
                                oData.availableAgents = data.agents || [];
                                oData.availableOperations = data.operations || [];
                                oModel.setData(oData);
                                resolve(data);
                            },
                            error(xhr) {
                                reject(new Error(`Failed to load available agents: ${ xhr.responseText}`));
                            }
                        });
                    }
                });
            }).catch((error) => {
                MessageBox.error(error.message);
            });
        },

        /**
         * @function onAgentDashboard
         * @description Opens agent management dashboard.
         * @public
         */
        onAgentDashboard() {
            this._getOrCreateDialog("agentDashboard", "a2a.network.agent7.ext.fragment.AgentDashboard")
                .then((oDialog) => {
                    oDialog.open();
                    this._loadDashboardData(oDialog);
                });
        },

        /**
         * @function _loadDashboardData
         * @description Loads dashboard data with metrics and charts.
         * @private
         */
        _loadDashboardData(oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["agentDashboard"];
            if (!oTargetDialog) {return;}

            oTargetDialog.setBusy(true);

            this._withErrorRecovery(() => {
                return new Promise((resolve, reject) => {
                    jQuery.ajax({
                        url: "/a2a/agent7/v1/dashboard",
                        type: "GET",
                        success(data) {
                            const oDashboardModel = new JSONModel({
                                summary: data.summary,
                                agentHealth: data.agentHealth,
                                performance: data.performance,
                                operations: data.operations,
                                alerts: data.alerts,
                                trends: data.trends
                            });

                            oTargetDialog.setModel(oDashboardModel, "dashboard");
                            resolve(data);
                        },
                        error(xhr) {
                            reject(new Error(`Failed to load dashboard data: ${ xhr.responseText}`));
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

        _createDashboardCharts(data, oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["agentDashboard"];
            if (!oTargetDialog) {return;}

            this._createAgentHealthChart(data.agentHealth, oTargetDialog);
            this._createPerformanceChart(data.performance, oTargetDialog);
            this._createOperationsChart(data.operations, oTargetDialog);
        },

        _createAgentHealthChart(healthData, oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["agentDashboard"];
            if (!oTargetDialog) {return;}

            const oVizFrame = oTargetDialog.byId("agentHealthChart");
            if (!oVizFrame || !healthData) {return;}

            const aChartData = healthData.map((agent) => {
                return {
                    Agent: agent.name,
                    Health: agent.healthScore,
                    Uptime: agent.uptime,
                    ResponseTime: agent.responseTime
                };
            });

            const oChartModel = new sap.ui.model.json.JSONModel({
                healthData: aChartData
            });
            oVizFrame.setModel(oChartModel);

            oVizFrame.setVizProperties({
                categoryAxis: {
                    title: { text: "Agents" }
                },
                valueAxis: {
                    title: { text: "Health Score" }
                },
                title: {
                    text: "Agent Health Overview"
                }
            });
        },

        /**
         * @function onRegisterAgent
         * @description Opens dialog to register new agent.
         * @public
         */
        onRegisterAgent() {
            // SECURITY FIX: Add authentication check for agent registration
            if (!this._hasRole("AgentAdmin")) {
                MessageBox.error("Access denied: Insufficient privileges to register new agents");
                this._logSecurityEvent("REGISTER_AGENT_ACCESS_DENIED", "User attempted agent registration without AgentAdmin role");
                return;
            }

            if (!this._authorizeOperation("UPDATE_AGENT")) {
                MessageBox.error("Access denied: Agent registration requires administrator privileges");
                this._logSecurityEvent("REGISTER_AGENT_UNAUTHORIZED", "Unauthorized agent registration attempt");
                return;
            }

            this._getOrCreateDialog("registerAgent", "a2a.network.agent7.ext.fragment.RegisterAgent")
                .then((oDialog) => {
                    const oModel = new JSONModel({
                        agentName: "",
                        agentType: "",
                        version: "",
                        endpoint: "",
                        port: 8000,
                        capabilities: [],
                        dependencies: [],
                        configuration: {},
                        autoStart: true
                    });
                    oDialog.setModel(oModel, "register");
                    oDialog.open();
                    this._loadAgentTypes(oDialog);

                    this._logSecurityEvent("REGISTER_AGENT_AUTHORIZED", "Agent registration dialog opened");
                })
                .catch((error) => {
                    MessageBox.error(`Failed to open agent registration dialog: ${ error.message}`);
                    this._logSecurityEvent("REGISTER_AGENT_ERROR", "Failed to open dialog", { error: error.message });
                });
        },

        _loadAgentTypes(oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["registerAgent"];
            if (!oTargetDialog) {return;}

            this._withErrorRecovery(() => {
                return new Promise((resolve, reject) => {
                    // SECURITY FIX: Use secure AJAX request instead of direct jQuery.ajax
                    if (this._securityUtils) {
                        this._securityUtils.secureAjaxRequest({
                            url: "/a2a/agent7/v1/agent-types",
                            type: "GET"
                        }).then((data) => {
                            const oModel = oTargetDialog.getModel("register");
                            const oData = oModel.getData();
                            // Sanitize received data
                            oData.availableTypes = data.types ? data.types.map((type) => {
                                return this._securityUtils.sanitizeInput(type);
                            }) : [];
                            oData.availableCapabilities = data.capabilities ? data.capabilities.map((cap) => {
                                return this._securityUtils.sanitizeInput(cap);
                            }) : [];
                            oModel.setData(oData);
                            resolve(data);
                        }).catch((xhr) => {
                            reject(new Error(`Failed to load agent types: ${ xhr.responseText || "Network error"}`));
                        });
                    } else {
                        // Fallback to regular AJAX with basic headers
                        jQuery.ajax({
                            url: "/a2a/agent7/v1/agent-types",
                            type: "GET",
                            headers: {
                                "X-Requested-With": "XMLHttpRequest"
                            },
                            success(data) {
                                const oModel = oTargetDialog.getModel("register");
                                const oData = oModel.getData();
                                oData.availableTypes = data.types || [];
                                oData.availableCapabilities = data.capabilities || [];
                                oModel.setData(oData);
                                resolve(data);
                            },
                            error(xhr) {
                                reject(new Error(`Failed to load agent types: ${ xhr.responseText}`));
                            }
                        });
                    }
                });
            }).catch((error) => {
                MessageBox.error(error.message);
            });
        },

        /**
         * @function onHealthMonitor
         * @description Opens real-time health monitoring interface.
         * @public
         */
        onHealthMonitor() {
            this._getOrCreateDialog("healthMonitor", "a2a.network.agent7.ext.fragment.HealthMonitor")
                .then((oDialog) => {
                    oDialog.open();
                    this._loadHealthData(oDialog);
                    this._startHealthMonitoring(oDialog);
                });
        },

        _loadHealthData(oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["healthMonitor"];
            if (!oTargetDialog) {return;}

            this._withErrorRecovery(() => {
                return new Promise((resolve, reject) => {
                    jQuery.ajax({
                        url: "/a2a/agent7/v1/health-status",
                        type: "GET",
                        success(data) {
                            const oModel = new JSONModel({
                                healthChecks: data.healthChecks,
                                systemHealth: data.systemHealth,
                                alerts: data.alerts,
                                recommendations: data.recommendations
                            });
                            oTargetDialog.setModel(oModel, "health");
                            resolve(data);
                        },
                        error(xhr) {
                            reject(new Error(`Failed to load health data: ${ xhr.responseText}`));
                        }
                    });
                });
            }).catch((error) => {
                MessageBox.error(error.message);
            });
        },

        _startHealthMonitoring(oDialog) {
            if (this._healthEventSource) {
                this._healthEventSource.close();
            }

            this._healthEventSource = new EventSource("/a2a/agent7/v1/health-stream");

            this._healthEventSource.onmessage = function(event) {
                const data = JSON.parse(event.data);
                this._updateHealthDisplay(data, oDialog);
            }.bind(this);

            this._healthEventSource.onerror = function() {
                const oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                const sErrorMsg = oBundle.getText("error.healthMonitoringLost") || "Health monitoring connection lost";
                MessageToast.show(sErrorMsg);
            }.bind(this);
        },

        _updateHealthDisplay(healthData, oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["healthMonitor"];
            if (!oTargetDialog) {return;}

            const oModel = oTargetDialog.getModel("health");
            if (!oModel) {return;}

            const oData = oModel.getData();

            if (healthData.type === "health_update") {
                const agentIndex = oData.healthChecks.findIndex((agent) => {
                    return agent.agentId === healthData.agentId;
                });

                if (agentIndex >= 0) {
                    oData.healthChecks[agentIndex] = healthData.health;
                    oModel.setData(oData);
                }
            } else if (healthData.type === "alert") {
                oData.alerts.unshift(healthData.alert);
                if (oData.alerts.length > 50) {
                    oData.alerts.pop();
                }
                oModel.setData(oData);

                if (healthData.alert.severity === "CRITICAL") {
                    MessageToast.show(`Critical Alert: ${ healthData.alert.message}`);
                }
            }
        },

        /**
         * @function onPerformanceAnalyzer
         * @description Opens performance analysis dialog.
         * @public
         */
        onPerformanceAnalyzer() {
            this._getOrCreateDialog("performanceAnalyzer", "a2a.network.agent7.ext.fragment.PerformanceAnalyzer")
                .then((oDialog) => {
                    oDialog.open();
                    this._loadPerformanceData(oDialog);
                });
        },

        _loadPerformanceData(oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["performanceAnalyzer"];
            if (!oTargetDialog) {return;}

            this._withErrorRecovery(() => {
                return new Promise((resolve, reject) => {
                    jQuery.ajax({
                        url: "/a2a/agent7/v1/performance-metrics",
                        type: "GET",
                        success(data) {
                            const oModel = new JSONModel({
                                metrics: data.metrics,
                                trends: data.trends,
                                benchmarks: data.benchmarks,
                                recommendations: data.recommendations
                            });
                            oTargetDialog.setModel(oModel, "performance");
                            resolve(data);
                        },
                        error(xhr) {
                            reject(new Error(`Failed to load performance data: ${ xhr.responseText}`));
                        }
                    });
                });
            }).then((data) => {
                this._createPerformanceCharts(data, oTargetDialog);
            }).catch((error) => {
                MessageBox.error(error.message);
            });
        },

        _createPerformanceCharts(data, oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["performanceAnalyzer"];
            if (!oTargetDialog) {return;}

            const oResponseTimeChart = oTargetDialog.byId("responseTimeChart");
            const oThroughputChart = oTargetDialog.byId("throughputChart");

            if (oResponseTimeChart && data.trends) {
                const aResponseData = data.trends.map((trend) => {
                    return {
                        Time: trend.timestamp,
                        ResponseTime: trend.averageResponseTime
                    };
                });

                const oResponseModel = new sap.ui.model.json.JSONModel({
                    responseData: aResponseData
                });
                oResponseTimeChart.setModel(oResponseModel);
            }

            if (oThroughputChart && data.trends) {
                const aThroughputData = data.trends.map((trend) => {
                    return {
                        Time: trend.timestamp,
                        Throughput: trend.throughput
                    };
                });

                const oThroughputModel = new sap.ui.model.json.JSONModel({
                    throughputData: aThroughputData
                });
                oThroughputChart.setModel(oThroughputModel);
            }
        },

        /**
         * @function onAgentCoordinator
         * @description Opens agent coordination management interface.
         * @public
         */
        onAgentCoordinator() {
            this._getOrCreateDialog("agentCoordinator", "a2a.network.agent7.ext.fragment.AgentCoordinator")
                .then((oDialog) => {
                    oDialog.open();
                    this._loadCoordinationData(oDialog);
                });
        },

        _loadCoordinationData(oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["agentCoordinator"];
            if (!oTargetDialog) {return;}

            this._withErrorRecovery(() => {
                return new Promise((resolve, reject) => {
                    jQuery.ajax({
                        url: "/a2a/agent7/v1/coordination-status",
                        type: "GET",
                        success(data) {
                            const oModel = new JSONModel({
                                activeConnections: data.activeConnections,
                                workflowStatus: data.workflowStatus,
                                loadBalancing: data.loadBalancing,
                                failoverStatus: data.failoverStatus
                            });
                            oTargetDialog.setModel(oModel, "coordination");
                            resolve(data);
                        },
                        error(xhr) {
                            reject(new Error(`Failed to load coordination data: ${ xhr.responseText}`));
                        }
                    });
                });
            }).catch((error) => {
                MessageBox.error(error.message);
            });
        },

        /**
         * @function onBulkOperations
         * @description Opens bulk operations dialog for selected agents.
         * @public
         */
        onBulkOperations() {
            // CRITICAL SECURITY FIX: Add authentication and authorization checks
            if (!this._hasRole("AgentAdmin")) {
                MessageBox.error("Access denied: Insufficient privileges for bulk operations");
                this._logSecurityEvent("BULK_OPERATIONS_ACCESS_DENIED", "User attempted bulk operations without AgentAdmin role", {
                    userRoles: this._getUserRoles(),
                    timestamp: new Date().toISOString()
                });
                return;
            }

            if (!this._authorizeOperation("BULK_OPERATIONS")) {
                MessageBox.error("Access denied: Bulk operations require administrator privileges");
                this._logSecurityEvent("BULK_OPERATIONS_UNAUTHORIZED", "Unauthorized bulk operations attempt", {
                    operation: "BULK_OPERATIONS",
                    authorized: false
                });
                return;
            }

            const oTable = this._extensionAPI.getTable();
            const aSelectedContexts = oTable.getSelectedContexts();

            if (aSelectedContexts.length === 0) {
                const oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                const sWarningMsg = oBundle.getText("msg.selectAgentsForBulk") || "Please select at least one agent for bulk operations.";
                MessageBox.warning(sWarningMsg);
                return;
            }

            // Security validation: Limit bulk operations to prevent resource exhaustion
            if (aSelectedContexts.length > 50) {
                MessageBox.error("Bulk operations limited to 50 agents maximum for security reasons");
                this._logSecurityEvent("BULK_OPERATIONS_LIMIT_EXCEEDED", "Bulk operation attempted on too many agents", {
                    selectedCount: aSelectedContexts.length,
                    maxAllowed: 50
                });
                return;
            }

            this._getOrCreateDialog("bulkOperations", "a2a.network.agent7.ext.fragment.BulkOperations")
                .then((oDialog) => {
                    let aAgentIds = aSelectedContexts.map((oContext) => {
                        const sAgentId = oContext.getProperty("managedAgent");
                        // Sanitize agent ID to prevent injection attacks
                        return this._securityUtils ? this._securityUtils.sanitizeInput(sAgentId) : sAgentId;
                    });

                    // Validate bulk operation data using SecurityUtils
                    if (this._securityUtils) {
                        const oBulkValidation = this._securityUtils.validateBulkOperation(aAgentIds, "BULK_MANAGE");
                        if (!oBulkValidation.isValid) {
                            MessageBox.error(`Bulk operation validation failed: ${ oBulkValidation.errors.join(", ")}`);
                            return;
                        }
                        aAgentIds = oBulkValidation.sanitizedIds;
                    }

                    const oModel = new JSONModel({
                        selectedAgents: aAgentIds,
                        operation: "",
                        parameters: {},
                        executeInParallel: true,
                        rollbackOnFailure: true
                    });
                    oDialog.setModel(oModel, "bulk");
                    oDialog.open();

                    // Log successful authorization
                    this._logSecurityEvent("BULK_OPERATIONS_AUTHORIZED", "Bulk operations dialog opened", {
                        agentCount: aAgentIds.length,
                        userRoles: this._getUserRoles()
                    });
                })
                .catch((error) => {
                    MessageBox.error(`Failed to open bulk operations dialog: ${ error.message}`);
                    this._logSecurityEvent("BULK_OPERATIONS_ERROR", "Failed to open dialog", { error: error.message });
                });
        },

        /**
         * @function _startRealtimeUpdates
         * @description Establishes EventSource connection for real-time updates.
         * @private
         */
        _startRealtimeUpdates() {
            this._realtimeEventSource = new EventSource("/a2a/agent7/v1/realtime-updates");

            this._realtimeEventSource.onmessage = function(event) {
                const data = JSON.parse(event.data);

                if (data.type === "agent_status_change") {
                    this._extensionAPI.refresh();
                    MessageToast.show(`Agent ${ data.agentName } status changed to ${ data.status}`);
                } else if (data.type === "performance_alert") {
                    MessageToast.show(`Performance Alert: ${ data.message}`);
                }
            }.bind(this);

            this._realtimeEventSource.onerror = function() {
                MessageToast.show("Real-time updates disconnected");
            }.bind(this);
        },

        /**
         * @function onConfirmCreateTask
         * @description Confirms and creates management task.
         * @public
         */
        onConfirmCreateTask() {
            const oDialog = this._dialogCache["createManagementTask"];
            if (!oDialog) {return;}

            const oModel = oDialog.getModel("create");
            const oData = oModel.getData();

            if (!this._validateInput(oData.taskName) || !this._validateInput(oData.managedAgent) ||
                !this._validateInput(oData.operationType)) {
                const oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                const sErrorMsg = oBundle.getText("validation.requiredFields") || "Please fill all required fields";
                MessageBox.error(sErrorMsg);
                return;
            }

            const sanitizedData = this._sanitizeObject(oData);
            oDialog.setBusy(true);

            this._withErrorRecovery(() => {
                return new Promise((resolve, reject) => {
                    jQuery.ajax({
                        url: "/a2a/agent7/v1/management-tasks",
                        type: "POST",
                        contentType: "application/json",
                        data: JSON.stringify(sanitizedData),
                        success(data) {
                            resolve(data);
                        },
                        error(xhr) {
                            reject(new Error(`Failed to create task: ${ xhr.responseText}`));
                        }
                    });
                });
            }).then((data) => {
                oDialog.setBusy(false);
                oDialog.close();
                const oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                const sSuccessMsg = oBundle.getText("success.taskCreated") || "Management task created successfully";
                MessageToast.show(sSuccessMsg);
                this._extensionAPI.refresh();
            }).catch((error) => {
                oDialog.setBusy(false);
                MessageBox.error(error.message);
            });
        },

        /**
         * @function onConfirmRegisterAgent
         * @description Confirms and registers new agent.
         * @public
         */
        onConfirmRegisterAgent() {
            const oModel = this._oRegisterDialog.getModel("register");
            const oData = oModel.getData();

            if (!this._validateInput(oData.agentName, "agentName")) {
                MessageBox.error("Invalid agent name. Use only alphanumeric characters, spaces, hyphens, and underscores (max 50 chars)");
                return;
            }

            if (!this._validateInput(oData.agentType)) {
                MessageBox.error("Invalid agent type");
                return;
            }

            if (!this._validateInput(oData.endpoint, "endpoint")) {
                MessageBox.error("Invalid endpoint URL format");
                return;
            }

            const sanitizedData = {
                agentName: this._sanitizeInput(oData.agentName),
                agentType: this._sanitizeInput(oData.agentType),
                version: this._sanitizeInput(oData.version),
                endpoint: this._sanitizeInput(oData.endpoint),
                port: Math.max(1, Math.min(65535, parseInt(oData.port, 10) || 8000)),
                capabilities: this._sanitizeArray(oData.capabilities),
                dependencies: this._sanitizeArray(oData.dependencies),
                configuration: this._sanitizeObject(oData.configuration),
                autoStart: Boolean(oData.autoStart)
            };

            this._oRegisterDialog.setBusy(true);

            this._secureAjaxCall({
                url: "/a2a/agent7/v1/register-agent",
                type: "POST",
                data: JSON.stringify(sanitizedData)
            }).then(result => {
                this._oRegisterDialog.setBusy(false);
                this._oRegisterDialog.close();

                const data = result.data;
                MessageBox.success(
                    "Agent registered successfully!\\n" +
                    `Agent ID: ${ this._sanitizeInput(data.agentId) }\\n` +
                    `Registration Block: ${ this._sanitizeInput(data.blockNumber)}`
                );
                this._extensionAPI.refresh();

                this._auditLogger.log("AGENT_REGISTERED", {
                    agentName: sanitizedData.agentName,
                    agentType: sanitizedData.agentType,
                    agentId: data.agentId
                });
            }).catch(error => {
                this._oRegisterDialog.setBusy(false);
                const errorMsg = this._sanitizeInput(error.xhr?.responseText || "Unknown error");
                MessageBox.error(`Agent registration failed: ${ errorMsg}`);
                this._auditLogger.log("AGENT_REGISTRATION_FAILED", { error: errorMsg });
            });
        },

        onCancelCreateTask() {
            const oDialog = this._dialogCache["createManagementTask"];
            if (oDialog) {
                oDialog.close();
            }
        },

        onCancelRegisterAgent() {
            const oDialog = this._dialogCache["registerAgent"];
            if (oDialog) {
                oDialog.close();
            }
        },

        _sanitizeObject(obj) {
            if (!obj || typeof obj !== "object") {return {};}
            const sanitized = {};
            Object.keys(obj).forEach(key => {
                if (typeof obj[key] === "string") {
                    sanitized[key] = this._sanitizeInput(obj[key]);
                } else if (Array.isArray(obj[key])) {
                    sanitized[key] = this._sanitizeArray(obj[key]);
                } else if (typeof obj[key] === "object") {
                    sanitized[key] = this._sanitizeObject(obj[key]);
                } else {
                    sanitized[key] = obj[key];
                }
            });
            return sanitized;
        },

        _sanitizeArray(arr) {
            if (!Array.isArray(arr)) {return [];}
            return arr.map(item => {
                if (typeof item === "string") {
                    return this._sanitizeInput(item);
                } else if (typeof item === "object") {
                    return this._sanitizeObject(item);
                }
                return item;

            });
        },

        /**
         * @function _validateInput
         * @description Validates input strings for security and format.
         * @param {string} input - Input to validate
         * @param {string} type - Validation type (optional)
         * @returns {boolean} True if valid
         * @private
         */
        _validateInput(input, type) {
            if (!input || typeof input !== "string") {return false;}

            switch (type) {
            case "agentName":
                return /^[a-zA-Z0-9\s\-_]{1,50}$/.test(input);
            case "endpoint":
                return /^https?:\/\/[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]+$/.test(input);
            default:
                return input.length > 0 && input.length <= 255;
            }
        },

        /**
         * @function _sanitizeInput
         * @description Sanitizes user input to prevent XSS attacks.
         * @param {string} input - Input to sanitize
         * @returns {string} Sanitized input
         * @private
         */
        _sanitizeInput(input) {
            if (!input) {return "";}
            return encodeXML(input.toString().trim());
        },

        /**
         * @function _createPerformanceChart
         * @description Creates performance visualization chart.
         * @param {Object} data - Performance data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createPerformanceChart(data, oDialog) {
            // Implementation placeholder for performance chart creation
            // This would create charts for CPU, memory, and other performance metrics
        },

        /**
         * @function _createOperationsChart
         * @description Creates operations visualization chart.
         * @param {Object} data - Operations data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createOperationsChart(data, oDialog) {
            // Implementation placeholder for operations chart creation
            // This would create charts for operation counts, success rates, etc.
        },

        /**
         * @function onCreateStreamProcessing
         * @description Creates new real-time stream processing task.
         * @public
         * @memberof a2a.network.agent7.ext.controller.ListReportExt
         * @since 1.0.0
         */
        onCreateStreamProcessing() {
            if (!this._hasRole("StreamProcessor")) {
                MessageBox.error("Access denied: Insufficient privileges for creating stream processing tasks");
                this._auditLogger.log("CREATE_STREAM_ACCESS_DENIED", { action: "create_stream_processing" });
                return;
            }

            this._getOrCreateDialog("createStream", "a2a.network.agent7.ext.fragment.CreateStreamProcessing")
                .then((oDialog) => {
                    const oModel = new JSONModel({
                        streamName: "",
                        description: "",
                        dataSource: {
                            type: "KAFKA",
                            endpoint: "",
                            topics: []
                        },
                        processingPipeline: {
                            filters: [],
                            transformations: [],
                            aggregations: []
                        },
                        outputTarget: {
                            type: "DATABASE",
                            endpoint: "",
                            format: "JSON"
                        },
                        realTimeConfig: {
                            batchSize: 1000,
                            windowSize: "5m",
                            watermarkDelay: "10s",
                            checkpointInterval: "30s"
                        }
                    });
                    oDialog.setModel(oModel, "stream");
                    oDialog.open();

                    this._auditLogger.log("CREATE_STREAM_INITIATED", { action: "create_stream_processing" });
                });
        },

        /**
         * @function onConfigureStreams
         * @description Opens stream configuration management interface.
         * @public
         * @memberof a2a.network.agent7.ext.controller.ListReportExt
         * @since 1.0.0
         */
        onConfigureStreams() {
            if (!this._hasRole("StreamProcessor")) {
                MessageBox.error("Access denied: Insufficient privileges for configuring streams");
                this._auditLogger.log("CONFIGURE_STREAMS_ACCESS_DENIED", { action: "configure_streams" });
                return;
            }

            this._getOrCreateDialog("configureStreams", "a2a.network.agent7.ext.fragment.StreamConfiguration")
                .then((oDialog) => {
                    oDialog.open();
                    this._loadStreamConfigurations(oDialog);

                    this._auditLogger.log("CONFIGURE_STREAMS_OPENED", { action: "configure_streams" });
                });
        },

        /**
         * @function onViewMetrics
         * @description Opens real-time metrics monitoring dashboard.
         * @public
         * @memberof a2a.network.agent7.ext.controller.ListReportExt
         * @since 1.0.0
         */
        onViewMetrics() {
            this._getOrCreateDialog("metricsViewer", "a2a.network.agent7.ext.fragment.RealTimeMetrics")
                .then((oDialog) => {
                    oDialog.open();
                    this._loadRealTimeMetrics(oDialog);
                    this._startMetricsStream(oDialog);

                    this._auditLogger.log("METRICS_VIEW_OPENED", { action: "view_metrics" });
                });
        },

        /**
         * @function _loadStreamConfigurations
         * @description Loads existing stream configurations.
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _loadStreamConfigurations(oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["configureStreams"];
            if (!oTargetDialog) {return;}

            oTargetDialog.setBusy(true);

            this._withErrorRecovery(function() {
                return new Promise((resolve, reject) => {
                    jQuery.ajax({
                        url: "/a2a/agent7/v1/stream-configurations",
                        type: "GET",
                        success: function(data) {
                            const oModel = new JSONModel({
                                streams: this._sanitizeArray(data.streams),
                                templates: this._sanitizeArray(data.templates),
                                connectionStatus: this._sanitizeObject(data.connectionStatus)
                            });
                            oTargetDialog.setModel(oModel, "config");
                            resolve(data);
                        }.bind(this),
                        error(xhr) {
                            reject(new Error(`Failed to load stream configurations: ${ xhr.responseText}`));
                        }
                    });
                });
            }).then((data) => {
                oTargetDialog.setBusy(false);
                this._auditLogger.log("STREAM_CONFIG_LOADED", { streamCount: data.streams.length });
            }).catch((error) => {
                oTargetDialog.setBusy(false);
                const errorMsg = this._sanitizeInput(error.message || "Unknown error");
                MessageBox.error(`Failed to load stream configurations: ${ errorMsg}`);
                this._auditLogger.log("STREAM_CONFIG_LOAD_FAILED", { error: errorMsg });
            });
        },

        /**
         * @function _loadRealTimeMetrics
         * @description Loads real-time processing metrics.
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _loadRealTimeMetrics(oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["metricsViewer"];
            if (!oTargetDialog) {return;}

            oTargetDialog.setBusy(true);

            this._withErrorRecovery(function() {
                return new Promise((resolve, reject) => {
                    jQuery.ajax({
                        url: "/a2a/agent7/v1/realtime-metrics",
                        type: "GET",
                        success: function(data) {
                            const oModel = new JSONModel({
                                throughput: this._sanitizeObject(data.throughput),
                                latency: this._sanitizeObject(data.latency),
                                errorRates: this._sanitizeObject(data.errorRates),
                                resourceUtilization: this._sanitizeObject(data.resourceUtilization),
                                streamHealth: this._sanitizeArray(data.streamHealth)
                            });
                            oTargetDialog.setModel(oModel, "metrics");
                            resolve(data);
                        }.bind(this),
                        error(xhr) {
                            reject(new Error(`Failed to load metrics: ${ xhr.responseText}`));
                        }
                    });
                });
            }).then((data) => {
                oTargetDialog.setBusy(false);
                this._createMetricsCharts(data, oTargetDialog);
                this._auditLogger.log("METRICS_LOADED", { action: "load_metrics" });
            }).catch((error) => {
                oTargetDialog.setBusy(false);
                const errorMsg = this._sanitizeInput(error.message || "Unknown error");
                MessageBox.error(`Failed to load metrics: ${ errorMsg}`);
                this._auditLogger.log("METRICS_LOAD_FAILED", { error: errorMsg });
            });
        },

        /**
         * @function _startMetricsStream
         * @description Starts real-time metrics streaming.
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _startMetricsStream(oDialog) {
            if (this._metricsEventSource) {
                this._metricsEventSource.close();
            }

            this._metricsEventSource = new EventSource("/a2a/agent7/v1/metrics-stream");

            this._metricsEventSource.onmessage = function(event) {
                const data = JSON.parse(event.data);
                this._updateMetricsDisplay(data, oDialog);
            }.bind(this);

            this._metricsEventSource.onerror = function() {
                const oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                const sErrorMsg = oBundle.getText("error.metricsStreamLost") || "Metrics streaming connection lost";
                MessageToast.show(sErrorMsg);
            }.bind(this);
        },

        /**
         * @function _updateMetricsDisplay
         * @description Updates metrics display with real-time data.
         * @param {Object} metricsData - Real-time metrics data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _updateMetricsDisplay(metricsData, oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["metricsViewer"];
            if (!oTargetDialog) {return;}

            const oModel = oTargetDialog.getModel("metrics");
            if (!oModel) {return;}

            const oData = oModel.getData();

            if (metricsData.type === "throughput_update") {
                oData.throughput = this._sanitizeObject(metricsData.data);
                oModel.setData(oData);
                this._updateThroughputChart(metricsData.data, oTargetDialog);
            } else if (metricsData.type === "latency_update") {
                oData.latency = this._sanitizeObject(metricsData.data);
                oModel.setData(oData);
                this._updateLatencyChart(metricsData.data, oTargetDialog);
            }
        },

        /**
         * @function _createMetricsCharts
         * @description Creates visualization charts for metrics data.
         * @param {Object} data - Metrics data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createMetricsCharts(data, oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["metricsViewer"];
            if (!oTargetDialog) {return;}

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
        _createThroughputChart(throughputData, oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["metricsViewer"];
            if (!oTargetDialog) {return;}

            const oChart = oTargetDialog.byId("throughputChart");
            if (!oChart || !throughputData) {return;}

            const aChartData = throughputData.streams.map((stream) => {
                return {
                    Stream: stream.name,
                    Throughput: stream.recordsPerSecond,
                    BytesPerSecond: stream.bytesPerSecond
                };
            });

            const oChartModel = new JSONModel({
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
        _updateThroughputChart(data, oDialog) {
            this._createThroughputChart(data, oDialog);
        },

        /**
         * @function _createLatencyChart
         * @description Creates latency visualization chart.
         * @param {Object} latencyData - Latency data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createLatencyChart(latencyData, oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["metricsViewer"];
            if (!oTargetDialog) {return;}

            const oChart = oTargetDialog.byId("latencyChart");
            if (!oChart || !latencyData) {return;}

            const aChartData = latencyData.streams.map((stream) => {
                return {
                    Stream: stream.name,
                    P95Latency: stream.p95LatencyMs,
                    P99Latency: stream.p99LatencyMs,
                    AvgLatency: stream.avgLatencyMs
                };
            });

            const oChartModel = new JSONModel({
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
        _updateLatencyChart(data, oDialog) {
            this._createLatencyChart(data, oDialog);
        },

        /**
         * @function _createErrorRateChart
         * @description Creates error rate visualization chart.
         * @param {Object} errorData - Error rate data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createErrorRateChart(errorData, oDialog) {
            const oTargetDialog = oDialog || this._dialogCache["metricsViewer"];
            if (!oTargetDialog) {return;}

            const oChart = oTargetDialog.byId("errorRateChart");
            if (!oChart || !errorData) {return;}

            const aChartData = errorData.streams.map((stream) => {
                return {
                    Stream: stream.name,
                    ErrorRate: stream.errorRate,
                    TotalErrors: stream.totalErrors
                };
            });

            const oChartModel = new JSONModel({
                errorData: aChartData
            });
            oChart.setModel(oChartModel);
        },

        /**
         * @function _auditLogger
         * @description Audit logging utility.
         * @private
         */
        _auditLogger: {
            log(action, details) {
                const user = sap.ushell?.Container?.getUser()?.getId() || "anonymous";
                const timestamp = new Date().toISOString();
                const logEntry = {
                    timestamp,
                    user,
                    agent: "Agent7_RealTimeProcessing",
                    action,
                    details
                };
                Log.info(`AUDIT: ${ JSON.stringify(logEntry)}`);
            }
        }
    });
});