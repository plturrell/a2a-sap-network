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
], function (ControllerExtension, Fragment, MessageBox, MessageToast, JSONModel, encodeXML, encodeURL, Log, SecurityUtils) {
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
            onInit: function () {
                this._extensionAPI = this.base.getExtensionAPI();
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
            onExit: function() {
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
        _initializeDeviceModel: function() {
            var oDeviceModel = new sap.ui.model.json.JSONModel(sap.ui.Device);
            this.base.getView().setModel(oDeviceModel, "device");
        },
        
        /**
         * @function _initializeDialogCache
         * @description Initializes dialog cache for performance.
         * @private
         */
        _initializeDialogCache: function() {
            this._dialogCache = {};
        },
        
        /**
         * @function _initializePerformanceOptimizations
         * @description Sets up performance optimization features.
         * @private
         */
        _initializePerformanceOptimizations: function() {
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
        _throttle: function(fn, limit) {
            var inThrottle;
            return function() {
                var args = arguments;
                var context = this;
                if (!inThrottle) {
                    fn.apply(context, args);
                    inThrottle = true;
                    setTimeout(function() { inThrottle = false; }, limit);
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
        _debounce: function(fn, delay) {
            var timeoutId;
            return function() {
                var context = this;
                var args = arguments;
                clearTimeout(timeoutId);
                timeoutId = setTimeout(function() {
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
        _performSearch: function(sQuery) {
            // Implement search logic
        },
        
        /**
         * @function _getOrCreateDialog
         * @description Gets cached dialog or creates new one.
         * @param {string} sDialogId - Dialog identifier
         * @param {string} sFragmentName - Fragment name
         * @returns {Promise<sap.m.Dialog>} Promise resolving to dialog
         * @private
         */
        _getOrCreateDialog: function(sDialogId, sFragmentName) {
            var that = this;
            
            if (this._dialogCache[sDialogId]) {
                return Promise.resolve(this._dialogCache[sDialogId]);
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
         * @description Optimizes dialog for current device.
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
        },
        
        /**
         * @function _withErrorRecovery
         * @description Wraps operation with error recovery.
         * @param {Function} fnOperation - Operation to execute
         * @param {Object} oOptions - Recovery options
         * @returns {Promise} Promise with error recovery
         * @private
         */
        _withErrorRecovery: function(fnOperation, oOptions) {
            var that = this;
            var oConfig = Object.assign({}, this._errorRecoveryConfig, oOptions);
            
            function attempt(retriesLeft, delay) {
                return fnOperation().catch(function(error) {
                    if (retriesLeft > 0) {
                        var oBundle = that.base.getView().getModel("i18n").getResourceBundle();
                        var sRetryMsg = oBundle.getText("recovery.retrying");
                        MessageToast.show(sRetryMsg);
                        
                        return new Promise(function(resolve) {
                            setTimeout(resolve, delay);
                        }).then(function() {
                            var nextDelay = oConfig.exponentialBackoff ? delay * 2 : delay;
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
        _cleanupResources: function() {
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
            Object.keys(this._dialogCache).forEach(function(key) {
                if (this._dialogCache[key]) {
                    this._dialogCache[key].destroy();
                }
            }.bind(this));
            this._dialogCache = {};
            
            // Clean up legacy dialog references
            var aDialogs = ["_oCreateDialog", "_oDashboard", "_oRegisterDialog", 
                           "_oHealthMonitor", "_oPerformanceDialog", "_oCoordinatorDialog", 
                           "_oBulkDialog"];
            aDialogs.forEach(function(sDialog) {
                if (this[sDialog]) {
                    this[sDialog].destroy();
                    this[sDialog] = null;
                }
            }.bind(this));
        },

        /**
         * @function onCreateManagementTask
         * @description Opens dialog to create new agent management task.
         * @public
         */
        onCreateManagementTask: function() {
            this._getOrCreateDialog("createManagementTask", "a2a.network.agent7.ext.fragment.CreateManagementTask")
                .then(function(oDialog) {
                    var oModel = new JSONModel({
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
                }.bind(this));
        },

        /**
         * @function _loadAvailableAgents
         * @description Loads available agents for selection.
         * @private
         */
        _loadAvailableAgents: function(oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["createManagementTask"];
            if (!oTargetDialog) return;
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    jQuery.ajax({
                        url: "/a2a/agent7/v1/registered-agents",
                        type: "GET",
                        success: function(data) {
                            var oModel = oTargetDialog.getModel("create");
                            var oData = oModel.getData();
                            oData.availableAgents = data.agents;
                            oData.availableOperations = data.operations;
                            oModel.setData(oData);
                            resolve(data);
                        },
                        error: function(xhr) {
                            reject(new Error("Failed to load available agents: " + xhr.responseText));
                        }
                    });
                });
            }).catch(function(error) {
                MessageBox.error(error.message);
            });
        },

        /**
         * @function onAgentDashboard
         * @description Opens agent management dashboard.
         * @public
         */
        onAgentDashboard: function() {
            this._getOrCreateDialog("agentDashboard", "a2a.network.agent7.ext.fragment.AgentDashboard")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadDashboardData(oDialog);
                }.bind(this));
        },

        /**
         * @function _loadDashboardData
         * @description Loads dashboard data with metrics and charts.
         * @private
         */
        _loadDashboardData: function(oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["agentDashboard"];
            if (!oTargetDialog) return;
            
            oTargetDialog.setBusy(true);
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    jQuery.ajax({
                        url: "/a2a/agent7/v1/dashboard",
                        type: "GET",
                        success: function(data) {
                            var oDashboardModel = new JSONModel({
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
                        error: function(xhr) {
                            reject(new Error("Failed to load dashboard data: " + xhr.responseText));
                        }
                    });
                });
            }).then(function(data) {
                oTargetDialog.setBusy(false);
                this._createDashboardCharts(data, oTargetDialog);
            }.bind(this)).catch(function(error) {
                oTargetDialog.setBusy(false);
                MessageBox.error(error.message);
            });
        },

        _createDashboardCharts: function(data, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["agentDashboard"];
            if (!oTargetDialog) return;
            
            this._createAgentHealthChart(data.agentHealth, oTargetDialog);
            this._createPerformanceChart(data.performance, oTargetDialog);
            this._createOperationsChart(data.operations, oTargetDialog);
        },

        _createAgentHealthChart: function(healthData, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["agentDashboard"];
            if (!oTargetDialog) return;
            
            var oVizFrame = oTargetDialog.byId("agentHealthChart");
            if (!oVizFrame || !healthData) return;
            
            var aChartData = healthData.map(function(agent) {
                return {
                    Agent: agent.name,
                    Health: agent.healthScore,
                    Uptime: agent.uptime,
                    ResponseTime: agent.responseTime
                };
            });
            
            var oChartModel = new sap.ui.model.json.JSONModel({
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
        onRegisterAgent: function() {
            this._getOrCreateDialog("registerAgent", "a2a.network.agent7.ext.fragment.RegisterAgent")
                .then(function(oDialog) {
                    var oModel = new JSONModel({
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
                }.bind(this));
        },

        _loadAgentTypes: function(oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["registerAgent"];
            if (!oTargetDialog) return;
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    jQuery.ajax({
                        url: "/a2a/agent7/v1/agent-types",
                        type: "GET",
                        success: function(data) {
                            var oModel = oTargetDialog.getModel("register");
                            var oData = oModel.getData();
                            oData.availableTypes = data.types;
                            oData.availableCapabilities = data.capabilities;
                            oModel.setData(oData);
                            resolve(data);
                        },
                        error: function(xhr) {
                            reject(new Error("Failed to load agent types: " + xhr.responseText));
                        }
                    });
                });
            }).catch(function(error) {
                MessageBox.error(error.message);
            });
        },

        /**
         * @function onHealthMonitor
         * @description Opens real-time health monitoring interface.
         * @public
         */
        onHealthMonitor: function() {
            this._getOrCreateDialog("healthMonitor", "a2a.network.agent7.ext.fragment.HealthMonitor")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadHealthData(oDialog);
                    this._startHealthMonitoring(oDialog);
                }.bind(this));
        },

        _loadHealthData: function(oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["healthMonitor"];
            if (!oTargetDialog) return;
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    jQuery.ajax({
                        url: "/a2a/agent7/v1/health-status",
                        type: "GET",
                        success: function(data) {
                            var oModel = new JSONModel({
                                healthChecks: data.healthChecks,
                                systemHealth: data.systemHealth,
                                alerts: data.alerts,
                                recommendations: data.recommendations
                            });
                            oTargetDialog.setModel(oModel, "health");
                            resolve(data);
                        },
                        error: function(xhr) {
                            reject(new Error("Failed to load health data: " + xhr.responseText));
                        }
                    });
                });
            }).catch(function(error) {
                MessageBox.error(error.message);
            });
        },

        _startHealthMonitoring: function(oDialog) {
            if (this._healthEventSource) {
                this._healthEventSource.close();
            }
            
            this._healthEventSource = new EventSource("/a2a/agent7/v1/health-stream");
            
            this._healthEventSource.onmessage = function(event) {
                var data = JSON.parse(event.data);
                this._updateHealthDisplay(data, oDialog);
            }.bind(this);
            
            this._healthEventSource.onerror = function() {
                var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                var sErrorMsg = oBundle.getText("error.healthMonitoringLost") || "Health monitoring connection lost";
                MessageToast.show(sErrorMsg);
            }.bind(this);
        },

        _updateHealthDisplay: function(healthData, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["healthMonitor"];
            if (!oTargetDialog) return;
            
            var oModel = oTargetDialog.getModel("health");
            if (!oModel) return;
            
            var oData = oModel.getData();
            
            if (healthData.type === "health_update") {
                var agentIndex = oData.healthChecks.findIndex(function(agent) {
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
                    MessageToast.show("Critical Alert: " + healthData.alert.message);
                }
            }
        },

        /**
         * @function onPerformanceAnalyzer
         * @description Opens performance analysis dialog.
         * @public
         */
        onPerformanceAnalyzer: function() {
            this._getOrCreateDialog("performanceAnalyzer", "a2a.network.agent7.ext.fragment.PerformanceAnalyzer")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadPerformanceData(oDialog);
                }.bind(this));
        },

        _loadPerformanceData: function(oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["performanceAnalyzer"];
            if (!oTargetDialog) return;
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    jQuery.ajax({
                        url: "/a2a/agent7/v1/performance-metrics",
                        type: "GET",
                        success: function(data) {
                            var oModel = new JSONModel({
                                metrics: data.metrics,
                                trends: data.trends,
                                benchmarks: data.benchmarks,
                                recommendations: data.recommendations
                            });
                            oTargetDialog.setModel(oModel, "performance");
                            resolve(data);
                        },
                        error: function(xhr) {
                            reject(new Error("Failed to load performance data: " + xhr.responseText));
                        }
                    });
                });
            }).then(function(data) {
                this._createPerformanceCharts(data, oTargetDialog);
            }.bind(this)).catch(function(error) {
                MessageBox.error(error.message);
            });
        },

        _createPerformanceCharts: function(data, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["performanceAnalyzer"];
            if (!oTargetDialog) return;
            
            var oResponseTimeChart = oTargetDialog.byId("responseTimeChart");
            var oThroughputChart = oTargetDialog.byId("throughputChart");
            
            if (oResponseTimeChart && data.trends) {
                var aResponseData = data.trends.map(function(trend) {
                    return {
                        Time: trend.timestamp,
                        ResponseTime: trend.averageResponseTime
                    };
                });
                
                var oResponseModel = new sap.ui.model.json.JSONModel({
                    responseData: aResponseData
                });
                oResponseTimeChart.setModel(oResponseModel);
            }
            
            if (oThroughputChart && data.trends) {
                var aThroughputData = data.trends.map(function(trend) {
                    return {
                        Time: trend.timestamp,
                        Throughput: trend.throughput
                    };
                });
                
                var oThroughputModel = new sap.ui.model.json.JSONModel({
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
        onAgentCoordinator: function() {
            this._getOrCreateDialog("agentCoordinator", "a2a.network.agent7.ext.fragment.AgentCoordinator")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadCoordinationData(oDialog);
                }.bind(this));
        },

        _loadCoordinationData: function(oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["agentCoordinator"];
            if (!oTargetDialog) return;
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    jQuery.ajax({
                        url: "/a2a/agent7/v1/coordination-status",
                        type: "GET",
                        success: function(data) {
                            var oModel = new JSONModel({
                                activeConnections: data.activeConnections,
                                workflowStatus: data.workflowStatus,
                                loadBalancing: data.loadBalancing,
                                failoverStatus: data.failoverStatus
                            });
                            oTargetDialog.setModel(oModel, "coordination");
                            resolve(data);
                        },
                        error: function(xhr) {
                            reject(new Error("Failed to load coordination data: " + xhr.responseText));
                        }
                    });
                });
            }).catch(function(error) {
                MessageBox.error(error.message);
            });
        },

        /**
         * @function onBulkOperations
         * @description Opens bulk operations dialog for selected agents.
         * @public
         */
        onBulkOperations: function() {
            var oTable = this._extensionAPI.getTable();
            var aSelectedContexts = oTable.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                var sWarningMsg = oBundle.getText("msg.selectAgentsForBulk") || "Please select at least one agent for bulk operations.";
                MessageBox.warning(sWarningMsg);
                return;
            }
            
            this._getOrCreateDialog("bulkOperations", "a2a.network.agent7.ext.fragment.BulkOperations")
                .then(function(oDialog) {
                    var aAgentIds = aSelectedContexts.map(function(oContext) {
                        return oContext.getProperty("managedAgent");
                    });
                    
                    var oModel = new JSONModel({
                        selectedAgents: aAgentIds,
                        operation: "",
                        parameters: {},
                        executeInParallel: true,
                        rollbackOnFailure: true
                    });
                    oDialog.setModel(oModel, "bulk");
                    oDialog.open();
                }.bind(this));
        },

        /**
         * @function _startRealtimeUpdates
         * @description Establishes EventSource connection for real-time updates.
         * @private
         */
        _startRealtimeUpdates: function() {
            this._realtimeEventSource = new EventSource("/a2a/agent7/v1/realtime-updates");
            
            this._realtimeEventSource.onmessage = function(event) {
                var data = JSON.parse(event.data);
                
                if (data.type === "agent_status_change") {
                    this._extensionAPI.refresh();
                    MessageToast.show("Agent " + data.agentName + " status changed to " + data.status);
                } else if (data.type === "performance_alert") {
                    MessageToast.show("Performance Alert: " + data.message);
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
        onConfirmCreateTask: function() {
            var oDialog = this._dialogCache["createManagementTask"];
            if (!oDialog) return;
            
            var oModel = oDialog.getModel("create");
            var oData = oModel.getData();
            
            if (!this._validateInput(oData.taskName) || !this._validateInput(oData.managedAgent) || !this._validateInput(oData.operationType)) {
                var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                var sErrorMsg = oBundle.getText("validation.requiredFields") || "Please fill all required fields";
                MessageBox.error(sErrorMsg);
                return;
            }
            
            var sanitizedData = this._sanitizeObject(oData);
            oDialog.setBusy(true);
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    jQuery.ajax({
                        url: "/a2a/agent7/v1/management-tasks",
                        type: "POST",
                        contentType: "application/json",
                        data: JSON.stringify(sanitizedData),
                        success: function(data) {
                            resolve(data);
                        },
                        error: function(xhr) {
                            reject(new Error("Failed to create task: " + xhr.responseText));
                        }
                    });
                });
            }).then(function(data) {
                oDialog.setBusy(false);
                oDialog.close();
                var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                var sSuccessMsg = oBundle.getText("success.taskCreated") || "Management task created successfully";
                MessageToast.show(sSuccessMsg);
                this._extensionAPI.refresh();
            }.bind(this)).catch(function(error) {
                oDialog.setBusy(false);
                MessageBox.error(error.message);
            });
        },

        /**
         * @function onConfirmRegisterAgent
         * @description Confirms and registers new agent.
         * @public
         */
        onConfirmRegisterAgent: function() {
            var oModel = this._oRegisterDialog.getModel("register");
            var oData = oModel.getData();
            
            if (!this._validateInput(oData.agentName, 'agentName')) {
                MessageBox.error("Invalid agent name. Use only alphanumeric characters, spaces, hyphens, and underscores (max 50 chars)");
                return;
            }
            
            if (!this._validateInput(oData.agentType)) {
                MessageBox.error("Invalid agent type");
                return;
            }
            
            if (!this._validateInput(oData.endpoint, 'endpoint')) {
                MessageBox.error("Invalid endpoint URL format");
                return;
            }
            
            const sanitizedData = {
                agentName: this._sanitizeInput(oData.agentName),
                agentType: this._sanitizeInput(oData.agentType),
                version: this._sanitizeInput(oData.version),
                endpoint: this._sanitizeInput(oData.endpoint),
                port: Math.max(1, Math.min(65535, parseInt(oData.port) || 8000)),
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
                    "Agent ID: " + this._sanitizeInput(data.agentId) + "\\n" +
                    "Registration Block: " + this._sanitizeInput(data.blockNumber)
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
                MessageBox.error("Agent registration failed: " + errorMsg);
                this._auditLogger.log("AGENT_REGISTRATION_FAILED", { error: errorMsg });
            });
        },

        onCancelCreateTask: function() {
            var oDialog = this._dialogCache["createManagementTask"];
            if (oDialog) {
                oDialog.close();
            }
        },

        onCancelRegisterAgent: function() {
            var oDialog = this._dialogCache["registerAgent"];
            if (oDialog) {
                oDialog.close();
            }
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
        
        /**
         * @function _validateInput
         * @description Validates input strings for security and format.
         * @param {string} input - Input to validate
         * @param {string} type - Validation type (optional)
         * @returns {boolean} True if valid
         * @private
         */
        _validateInput: function(input, type) {
            if (!input || typeof input !== 'string') return false;
            
            switch(type) {
                case 'agentName':
                    return /^[a-zA-Z0-9\s\-_]{1,50}$/.test(input);
                case 'endpoint':
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
        _sanitizeInput: function(input) {
            if (!input) return "";
            return encodeXML(input.toString().trim());
        },
        
        /**
         * @function _sanitizeObject
         * @description Recursively sanitizes object properties.
         * @param {Object} obj - Object to sanitize
         * @returns {Object} Sanitized object
         * @private
         */
        _sanitizeObject: function(obj) {
            if (!obj || typeof obj !== 'object') return {};
            const sanitized = {};
            Object.keys(obj).forEach(function(key) {
                if (typeof obj[key] === 'string') {
                    sanitized[key] = this._sanitizeInput(obj[key]);
                } else if (Array.isArray(obj[key])) {
                    sanitized[key] = this._sanitizeArray(obj[key]);
                } else if (typeof obj[key] === 'object') {
                    sanitized[key] = this._sanitizeObject(obj[key]);
                } else {
                    sanitized[key] = obj[key];
                }
            }.bind(this));
            return sanitized;
        },
        
        /**
         * @function _sanitizeArray
         * @description Sanitizes array elements.
         * @param {Array} arr - Array to sanitize
         * @returns {Array} Sanitized array
         * @private
         */
        _sanitizeArray: function(arr) {
            if (!Array.isArray(arr)) return [];
            return arr.map(function(item) {
                if (typeof item === 'string') {
                    return this._sanitizeInput(item);
                } else if (typeof item === 'object') {
                    return this._sanitizeObject(item);
                } else {
                    return item;
                }
            }.bind(this));
        },
        
        /**
         * @function _createPerformanceChart
         * @description Creates performance visualization chart.
         * @param {Object} data - Performance data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createPerformanceChart: function(data, oDialog) {
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
        _createOperationsChart: function(data, oDialog) {
            // Implementation placeholder for operations chart creation
            // This would create charts for operation counts, success rates, etc.
        },
        
        /**
         * @function _startRealtimeUpdates
         * @description Establishes EventSource connection for real-time updates.
         * @private
         */
        _startRealtimeUpdates: function() {
            if (this._realtimeEventSource) {
                this._realtimeEventSource.close();
            }
            
            this._realtimeEventSource = new EventSource("/a2a/agent7/v1/realtime-updates");
            
            this._realtimeEventSource.onmessage = function(event) {
                var data = JSON.parse(event.data);
                
                if (data.type === "agent_status_change") {
                    this._throttledDashboardUpdate();
                    var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                    var sStatusMsg = oBundle.getText("msg.agentStatusChanged") || "Agent status changed";
                    MessageToast.show(sStatusMsg.replace("{0}", data.agentName).replace("{1}", data.status));
                } else if (data.type === "performance_alert") {
                    var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                    var sAlertMsg = oBundle.getText("msg.performanceAlert") || "Performance Alert";
                    MessageToast.show(sAlertMsg + ": " + data.message);
                }
            }.bind(this);
            
            this._realtimeEventSource.onerror = function() {
                var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                var sErrorMsg = oBundle.getText("error.realtimeDisconnected") || "Real-time updates disconnected";
                MessageToast.show(sErrorMsg);
            }.bind(this);
        },

        /**
         * @function onCreateStreamProcessing
         * @description Creates new real-time stream processing task.
         * @public
         * @memberof a2a.network.agent7.ext.controller.ListReportExt
         * @since 1.0.0
         */
        onCreateStreamProcessing: function() {
            if (!this._hasRole("StreamProcessor")) {
                MessageBox.error("Access denied: Insufficient privileges for creating stream processing tasks");
                this._auditLogger.log("CREATE_STREAM_ACCESS_DENIED", { action: "create_stream_processing" });
                return;
            }
            
            this._getOrCreateDialog("createStream", "a2a.network.agent7.ext.fragment.CreateStreamProcessing")
                .then(function(oDialog) {
                    var oModel = new JSONModel({
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
                }.bind(this));
        },

        /**
         * @function onConfigureStreams
         * @description Opens stream configuration management interface.
         * @public
         * @memberof a2a.network.agent7.ext.controller.ListReportExt
         * @since 1.0.0
         */
        onConfigureStreams: function() {
            if (!this._hasRole("StreamProcessor")) {
                MessageBox.error("Access denied: Insufficient privileges for configuring streams");
                this._auditLogger.log("CONFIGURE_STREAMS_ACCESS_DENIED", { action: "configure_streams" });
                return;
            }
            
            this._getOrCreateDialog("configureStreams", "a2a.network.agent7.ext.fragment.StreamConfiguration")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadStreamConfigurations(oDialog);
                    
                    this._auditLogger.log("CONFIGURE_STREAMS_OPENED", { action: "configure_streams" });
                }.bind(this));
        },

        /**
         * @function onViewMetrics
         * @description Opens real-time metrics monitoring dashboard.
         * @public
         * @memberof a2a.network.agent7.ext.controller.ListReportExt
         * @since 1.0.0
         */
        onViewMetrics: function() {
            this._getOrCreateDialog("metricsViewer", "a2a.network.agent7.ext.fragment.RealTimeMetrics")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadRealTimeMetrics(oDialog);
                    this._startMetricsStream(oDialog);
                    
                    this._auditLogger.log("METRICS_VIEW_OPENED", { action: "view_metrics" });
                }.bind(this));
        },

        /**
         * @function _loadStreamConfigurations
         * @description Loads existing stream configurations.
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _loadStreamConfigurations: function(oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["configureStreams"];
            if (!oTargetDialog) return;
            
            oTargetDialog.setBusy(true);
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    jQuery.ajax({
                        url: "/a2a/agent7/v1/stream-configurations",
                        type: "GET",
                        success: function(data) {
                            var oModel = new JSONModel({
                                streams: this._sanitizeArray(data.streams),
                                templates: this._sanitizeArray(data.templates),
                                connectionStatus: this._sanitizeObject(data.connectionStatus)
                            });
                            oTargetDialog.setModel(oModel, "config");
                            resolve(data);
                        }.bind(this),
                        error: function(xhr) {
                            reject(new Error("Failed to load stream configurations: " + xhr.responseText));
                        }
                    });
                }.bind(this));
            }).then(function(data) {
                oTargetDialog.setBusy(false);
                this._auditLogger.log("STREAM_CONFIG_LOADED", { streamCount: data.streams.length });
            }.bind(this)).catch(function(error) {
                oTargetDialog.setBusy(false);
                var errorMsg = this._sanitizeInput(error.message || "Unknown error");
                MessageBox.error("Failed to load stream configurations: " + errorMsg);
                this._auditLogger.log("STREAM_CONFIG_LOAD_FAILED", { error: errorMsg });
            }.bind(this));
        },

        /**
         * @function _loadRealTimeMetrics
         * @description Loads real-time processing metrics.
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _loadRealTimeMetrics: function(oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["metricsViewer"];
            if (!oTargetDialog) return;
            
            oTargetDialog.setBusy(true);
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    jQuery.ajax({
                        url: "/a2a/agent7/v1/realtime-metrics",
                        type: "GET",
                        success: function(data) {
                            var oModel = new JSONModel({
                                throughput: this._sanitizeObject(data.throughput),
                                latency: this._sanitizeObject(data.latency),
                                errorRates: this._sanitizeObject(data.errorRates),
                                resourceUtilization: this._sanitizeObject(data.resourceUtilization),
                                streamHealth: this._sanitizeArray(data.streamHealth)
                            });
                            oTargetDialog.setModel(oModel, "metrics");
                            resolve(data);
                        }.bind(this),
                        error: function(xhr) {
                            reject(new Error("Failed to load metrics: " + xhr.responseText));
                        }
                    });
                }.bind(this));
            }).then(function(data) {
                oTargetDialog.setBusy(false);
                this._createMetricsCharts(data, oTargetDialog);
                this._auditLogger.log("METRICS_LOADED", { action: "load_metrics" });
            }.bind(this)).catch(function(error) {
                oTargetDialog.setBusy(false);
                var errorMsg = this._sanitizeInput(error.message || "Unknown error");
                MessageBox.error("Failed to load metrics: " + errorMsg);
                this._auditLogger.log("METRICS_LOAD_FAILED", { error: errorMsg });
            }.bind(this));
        },

        /**
         * @function _startMetricsStream
         * @description Starts real-time metrics streaming.
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _startMetricsStream: function(oDialog) {
            if (this._metricsEventSource) {
                this._metricsEventSource.close();
            }
            
            this._metricsEventSource = new EventSource("/a2a/agent7/v1/metrics-stream");
            
            this._metricsEventSource.onmessage = function(event) {
                var data = JSON.parse(event.data);
                this._updateMetricsDisplay(data, oDialog);
            }.bind(this);
            
            this._metricsEventSource.onerror = function() {
                var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                var sErrorMsg = oBundle.getText("error.metricsStreamLost") || "Metrics streaming connection lost";
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
        _updateMetricsDisplay: function(metricsData, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["metricsViewer"];
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
            }
        },

        /**
         * @function _createMetricsCharts
         * @description Creates visualization charts for metrics data.
         * @param {Object} data - Metrics data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createMetricsCharts: function(data, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["metricsViewer"];
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
            var oTargetDialog = oDialog || this._dialogCache["metricsViewer"];
            if (!oTargetDialog) return;
            
            var oChart = oTargetDialog.byId("throughputChart");
            if (!oChart || !throughputData) return;
            
            var aChartData = throughputData.streams.map(function(stream) {
                return {
                    Stream: stream.name,
                    Throughput: stream.recordsPerSecond,
                    BytesPerSecond: stream.bytesPerSecond
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
            var oTargetDialog = oDialog || this._dialogCache["metricsViewer"];
            if (!oTargetDialog) return;
            
            var oChart = oTargetDialog.byId("latencyChart");
            if (!oChart || !latencyData) return;
            
            var aChartData = latencyData.streams.map(function(stream) {
                return {
                    Stream: stream.name,
                    P95Latency: stream.p95LatencyMs,
                    P99Latency: stream.p99LatencyMs,
                    AvgLatency: stream.avgLatencyMs
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
            var oTargetDialog = oDialog || this._dialogCache["metricsViewer"];
            if (!oTargetDialog) return;
            
            var oChart = oTargetDialog.byId("errorRateChart");
            if (!oChart || !errorData) return;
            
            var aChartData = errorData.streams.map(function(stream) {
                return {
                    Stream: stream.name,
                    ErrorRate: stream.errorRate,
                    TotalErrors: stream.totalErrors
                };
            });
            
            var oChartModel = new JSONModel({
                errorData: aChartData
            });
            oChart.setModel(oChartModel);
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
            return user && user.hasRole && user.hasRole(role);
        },

        /**
         * @function _auditLogger
         * @description Audit logging utility.
         * @private
         */
        _auditLogger: {
            log: function(action, details) {
                const user = sap.ushell?.Container?.getUser()?.getId() || "anonymous";
                const timestamp = new Date().toISOString();
                const logEntry = {
                    timestamp: timestamp,
                    user: user,
                    agent: "Agent7_RealTimeProcessing",
                    action: action,
                    details: details
                };
                Log.info("AUDIT: " + JSON.stringify(logEntry));
            }
        }
    });
});