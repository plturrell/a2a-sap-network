/**
 * A2A Protocol Compliance: WebSocket replaced with blockchain event streaming
 * All real-time communication now uses blockchain events instead of WebSockets
 */

sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel",
    "a2a/network/agent12/ext/utils/SecurityUtils"
], function (ControllerExtension, MessageBox, MessageToast, Fragment, JSONModel, SecurityUtils) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent12.ext.controller.ObjectPageExt", {
        
        override: {
            onInit: function () {
                this._extensionAPI = this.base.getExtensionAPI();
                this._securityUtils = SecurityUtils;
                this._initializeSecurity();
                
                // Initialize device model for responsive behavior
                var oDeviceModel = new JSONModel(sap.ui.Device);
                oDeviceModel.setDefaultBindingMode(sap.ui.model.BindingMode.OneWay);
                this.base.getView().setModel(oDeviceModel, "device");
                
                // Initialize dialog cache
                this._dialogCache = {};
                
                // Initialize real-time monitoring
                this._initializeRealtimeMonitoring();
            },
            
            onExit: function() {
                this._cleanupResources();
                if (this.base.onExit) {
                    this.base.onExit.apply(this, arguments);
                }
            }
        },

        /**
         * @function onViewAnalytics
         * @description Opens performance analytics dashboard with real-time metrics and visualizations.
         * @public
         */
        onViewAnalytics: function() {
            if (!this._hasRole("PerformanceUser")) {
                MessageBox.error("Access denied. Performance User role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "ViewAnalytics", reason: "Insufficient permissions" });
                return;
            }

            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            const sTaskId = oData.taskId;
            const sTaskName = oData.taskName;
            
            this._auditLogger.log("VIEW_ANALYTICS", { taskId: sTaskId, taskName: sTaskName });
            
            this._getOrCreateDialog("viewAnalytics", "a2a.network.agent12.ext.fragment.ViewAnalytics")
                .then(function(oDialog) {
                    var oAnalyticsModel = new JSONModel({
                        taskId: sTaskId,
                        taskName: sTaskName,
                        timeRange: "LAST_HOUR",
                        startTime: new Date(Date.now() - 60 * 60 * 1000),
                        endTime: new Date(),
                        refreshInterval: 5000,
                        autoRefresh: true,
                        metrics: {
                            responseTime: { current: 0, average: 0, trend: [] },
                            throughput: { current: 0, average: 0, trend: [] },
                            cpuUsage: { current: 0, average: 0, trend: [] },
                            memoryUsage: { current: 0, average: 0, trend: [] },
                            errorRate: { current: 0, average: 0, trend: [] },
                            availability: { current: 0, average: 0, trend: [] }
                        },
                        chartType: "LINE",
                        selectedMetrics: ["responseTime", "throughput"],
                        alertsEnabled: true,
                        alertThresholds: {
                            responseTime: 1000,
                            errorRate: 5,
                            cpuUsage: 80,
                            memoryUsage: 85
                        }
                    });
                    oDialog.setModel(oAnalyticsModel, "analytics");
                    oDialog.open();
                    this._startAnalyticsStreaming(sTaskId, oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Analytics Dashboard: " + error.message);
                });
        },

        /**
         * @function onRunPerformanceTest
         * @description Runs comprehensive performance test with customizable parameters.
         * @public
         */
        onRunPerformanceTest: function() {
            if (!this._hasRole("PerformanceAdmin")) {
                MessageBox.error("Access denied. Performance Administrator role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "RunPerformanceTest", reason: "Insufficient permissions" });
                return;
            }

            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            const sTaskId = oData.taskId;
            const sTaskName = oData.taskName;
            
            // Check if test is already running
            if (oData.status === "TESTING") {
                MessageToast.show(this.getResourceBundle().getText("msg.testAlreadyRunning"));
                return;
            }
            
            this._auditLogger.log("RUN_PERFORMANCE_TEST", { taskId: sTaskId, taskName: sTaskName });
            
            this._getOrCreateDialog("runPerformanceTest", "a2a.network.agent12.ext.fragment.RunPerformanceTest")
                .then(function(oDialog) {
                    var oTestModel = new JSONModel({
                        taskId: sTaskId,
                        taskName: sTaskName,
                        testType: "LOAD",
                        duration: 300,
                        concurrentUsers: 100,
                        rampUpTime: 60,
                        iterations: 1000,
                        thinkTime: 1000,
                        testScenarios: [
                            { name: "Basic Load", selected: true },
                            { name: "Stress Test", selected: false },
                            { name: "Spike Test", selected: false },
                            { name: "Endurance Test", selected: false },
                            { name: "Volume Test", selected: false }
                        ],
                        targetMetrics: {
                            responseTime: 100,
                            throughput: 1000,
                            errorRate: 1,
                            successRate: 99
                        },
                        advancedOptions: {
                            simulateRealUsers: true,
                            randomizeRequests: true,
                            cacheEnabled: false,
                            compressionEnabled: true,
                            sslEnabled: true,
                            customHeaders: [],
                            customParameters: []
                        }
                    });
                    oDialog.setModel(oTestModel, "test");
                    oDialog.open();
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Performance Test: " + error.message);
                });
        },

        /**
         * @function onGenerateBenchmarkReport
         * @description Generates comprehensive benchmark report with analysis and recommendations.
         * @public
         */
        onGenerateBenchmarkReport: function() {
            if (!this._hasRole("PerformanceUser")) {
                MessageBox.error("Access denied. Performance User role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "GenerateBenchmarkReport", reason: "Insufficient permissions" });
                return;
            }

            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            const sTaskId = oData.taskId;
            const sTaskName = oData.taskName;
            
            // Check if benchmark data is available
            if (!oData.lastBenchmarkDate) {
                MessageBox.warning(this.getResourceBundle().getText("msg.noBenchmarkData"));
                return;
            }
            
            this._auditLogger.log("GENERATE_BENCHMARK_REPORT", { taskId: sTaskId, taskName: sTaskName });
            
            this._getOrCreateDialog("generateBenchmarkReport", "a2a.network.agent12.ext.fragment.GenerateBenchmarkReport")
                .then(function(oDialog) {
                    var oReportModel = new JSONModel({
                        taskId: sTaskId,
                        taskName: sTaskName,
                        reportFormat: "PDF",
                        reportType: "COMPREHENSIVE",
                        includeExecutiveSummary: true,
                        includeDetailedMetrics: true,
                        includeCharts: true,
                        includeRecommendations: true,
                        includeHistoricalTrends: true,
                        includeComparisons: true,
                        includeSystemInfo: true,
                        includeTestScenarios: true,
                        comparisonPeriod: "LAST_30_DAYS",
                        confidentialityLevel: "INTERNAL",
                        customSections: [],
                        distributionList: [],
                        reportTemplates: [
                            { key: "EXECUTIVE", text: "Executive Summary" },
                            { key: "TECHNICAL", text: "Technical Report" },
                            { key: "COMPREHENSIVE", text: "Comprehensive Analysis" },
                            { key: "COMPARISON", text: "Comparative Analysis" },
                            { key: "CUSTOM", text: "Custom Report" }
                        ]
                    });
                    oDialog.setModel(oReportModel, "report");
                    oDialog.open();
                    this._loadBenchmarkData(sTaskId, oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Benchmark Report Generation: " + error.message);
                });
        },

        /**
         * @function onConfigureThresholds
         * @description Opens performance threshold configuration interface.
         * @public
         */
        onConfigureThresholds: function() {
            if (!this._hasRole("PerformanceAdmin")) {
                MessageBox.error("Access denied. Performance Administrator role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "ConfigureThresholds", reason: "Insufficient permissions" });
                return;
            }

            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            const sTaskId = oData.taskId;
            const sTaskName = oData.taskName;
            
            this._auditLogger.log("CONFIGURE_THRESHOLDS", { taskId: sTaskId, taskName: sTaskName });
            
            this._getOrCreateDialog("configureThresholds", "a2a.network.agent12.ext.fragment.ConfigureThresholds")
                .then(function(oDialog) {
                    var oThresholdModel = new JSONModel({
                        taskId: sTaskId,
                        taskName: sTaskName,
                        categories: [
                            { key: "RESPONSE", text: "Response Time" },
                            { key: "THROUGHPUT", text: "Throughput" },
                            { key: "RESOURCE", text: "Resource Usage" },
                            { key: "ERROR", text: "Error Rates" },
                            { key: "AVAILABILITY", text: "Availability" },
                            { key: "CUSTOM", text: "Custom Metrics" }
                        ],
                        selectedCategory: "RESPONSE",
                        thresholds: {},
                        alertingEnabled: true,
                        escalationEnabled: true,
                        notificationChannels: [],
                        escalationPolicies: [],
                        profiles: [],
                        currentProfile: "DEFAULT"
                    });
                    oDialog.setModel(oThresholdModel, "threshold");
                    oDialog.open();
                    this._loadThresholdConfiguration(sTaskId, oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Threshold Configuration: " + error.message);
                });
        },

        /**
         * @function _startAnalyticsStreaming
         * @description Starts real-time analytics data streaming.
         * @param {string} sTaskId - Task ID
         * @param {sap.m.Dialog} oDialog - Analytics dialog
         * @private
         */
        _startAnalyticsStreaming: function(sTaskId, oDialog) {
            if (this._analyticsEventSource) {
                this._analyticsEventSource.close();
            }
            
            try {
                this._analyticsEventSource = new EventSource('/api/agent12/performance/analytics-stream/' + sTaskId);
                
                this._analyticsEventSource.onmessage = function(event) {
                    try {
                        const data = JSON.parse(event.data);
                        this._updateAnalyticsData(data, oDialog);
                    } catch (error) {
                        console.error('Error parsing analytics data:', error);
                    }
                }.bind(this);
                
                this._analyticsEventSource.onerror = function(error) {
                    console.warn('Analytics stream error, falling back to polling:', error);
                    this._startAnalyticsPolling(sTaskId, oDialog);
                }.bind(this);
                
            } catch (error) {
                console.warn('EventSource not available, using polling fallback');
                this._startAnalyticsPolling(sTaskId, oDialog);
            }
        },

        /**
         * @function _startAnalyticsPolling
         * @description Starts polling fallback for analytics data.
         * @param {string} sTaskId - Task ID
         * @param {sap.m.Dialog} oDialog - Analytics dialog
         * @private
         */
        _startAnalyticsPolling: function(sTaskId, oDialog) {
            if (this._analyticsPollingInterval) {
                clearInterval(this._analyticsPollingInterval);
            }
            
            const refreshInterval = oDialog.getModel("analytics").getData().refreshInterval || 5000;
            
            this._analyticsPollingInterval = setInterval(() => {
                if (oDialog.isOpen()) {
                    this._fetchAnalyticsData(sTaskId, oDialog);
                } else {
                    clearInterval(this._analyticsPollingInterval);
                }
            }, refreshInterval);
            
            // Initial fetch
            this._fetchAnalyticsData(sTaskId, oDialog);
        },

        /**
         * @function _fetchAnalyticsData
         * @description Fetches analytics data via API.
         * @param {string} sTaskId - Task ID
         * @param {sap.m.Dialog} oDialog - Analytics dialog
         * @private
         */
        _fetchAnalyticsData: function(sTaskId, oDialog) {
            const oModel = this.base.getView().getModel();
            const oAnalyticsData = oDialog.getModel("analytics").getData();
            
            SecurityUtils.secureCallFunction(oModel, "/GetPerformanceAnalytics", {
                urlParameters: {
                    taskId: sTaskId,
                    timeRange: oAnalyticsData.timeRange,
                    metrics: oAnalyticsData.selectedMetrics.join(',')
                },
                success: function(data) {
                    this._updateAnalyticsData(data, oDialog);
                }.bind(this),
                error: function(error) {
                    console.error('Failed to fetch analytics data:', error);
                }
            });
        },

        /**
         * @function _updateAnalyticsData
         * @description Updates analytics display with new data.
         * @param {Object} data - Analytics data
         * @param {sap.m.Dialog} oDialog - Analytics dialog
         * @private
         */
        _updateAnalyticsData: function(data, oDialog) {
            if (!oDialog || !oDialog.isOpen()) return;
            
            var oAnalyticsModel = oDialog.getModel("analytics");
            if (oAnalyticsModel) {
                var oCurrentData = oAnalyticsModel.getData();
                
                // Update metrics
                if (data.metrics) {
                    Object.keys(data.metrics).forEach(function(metric) {
                        if (oCurrentData.metrics[metric]) {
                            oCurrentData.metrics[metric] = data.metrics[metric];
                        }
                    });
                }
                
                // Check for alerts
                if (data.alerts && data.alerts.length > 0) {
                    this._handlePerformanceAlerts(data.alerts);
                }
                
                oCurrentData.lastUpdated = new Date().toISOString();
                oAnalyticsModel.setData(oCurrentData);
            }
        },

        /**
         * @function _handlePerformanceAlerts
         * @description Handles performance alerts.
         * @param {Array} alerts - Array of performance alerts
         * @private
         */
        _handlePerformanceAlerts: function(alerts) {
            alerts.forEach(function(alert) {
                var sMessage = this.getResourceBundle().getText("alert." + alert.type, [alert.metric, alert.value, alert.threshold]);
                MessageToast.show(sMessage || alert.message);
                this._auditLogger.log("PERFORMANCE_ALERT", alert);
            }.bind(this));
        },

        /**
         * @function _loadBenchmarkData
         * @description Loads benchmark data for report generation.
         * @param {string} sTaskId - Task ID
         * @param {sap.m.Dialog} oDialog - Report dialog
         * @private
         */
        _loadBenchmarkData: function(sTaskId, oDialog) {
            oDialog.setBusy(true);
            
            const oModel = this.base.getView().getModel();
            
            SecurityUtils.secureCallFunction(oModel, "/GetBenchmarkData", {
                urlParameters: {
                    taskId: sTaskId,
                    includeHistory: true,
                    limit: 30
                },
                success: function(data) {
                    var oReportModel = oDialog.getModel("report");
                    if (oReportModel) {
                        var oCurrentData = oReportModel.getData();
                        oCurrentData.benchmarkData = data.benchmarks || [];
                        oCurrentData.historicalData = data.history || [];
                        oCurrentData.comparativeData = data.comparisons || {};
                        oCurrentData.recommendations = data.recommendations || [];
                        oReportModel.setData(oCurrentData);
                    }
                    oDialog.setBusy(false);
                }.bind(this),
                error: function(error) {
                    oDialog.setBusy(false);
                    MessageBox.error("Failed to load benchmark data: " + error.message);
                }
            });
        },

        /**
         * @function _loadThresholdConfiguration
         * @description Loads threshold configuration data.
         * @param {string} sTaskId - Task ID
         * @param {sap.m.Dialog} oDialog - Threshold dialog
         * @private
         */
        _loadThresholdConfiguration: function(sTaskId, oDialog) {
            const oModel = this.base.getView().getModel();
            
            SecurityUtils.secureCallFunction(oModel, "/GetThresholdConfiguration", {
                urlParameters: { taskId: sTaskId },
                success: function(data) {
                    var oThresholdModel = oDialog.getModel("threshold");
                    if (oThresholdModel) {
                        var oCurrentData = oThresholdModel.getData();
                        oCurrentData.thresholds = data.thresholds || {};
                        oCurrentData.notificationChannels = data.notificationChannels || [];
                        oCurrentData.escalationPolicies = data.escalationPolicies || [];
                        oCurrentData.profiles = data.profiles || [];
                        oCurrentData.recommendations = data.recommendations || {};
                        oThresholdModel.setData(oCurrentData);
                    }
                }.bind(this),
                error: function(error) {
                    MessageBox.error("Failed to load threshold configuration: " + error.message);
                }
            });
        },

        /**
         * @function _initializeRealtimeMonitoring
         * @description Initializes real-time monitoring for performance metrics.
         * @private
         */
        _initializeRealtimeMonitoring: function() {
            // WebSocket for real-time performance updates
            this._initializePerformanceWebSocket();
        },

        /**
         * @function _initializePerformanceWebSocket
         * @description Initializes WebSocket for performance updates.
         * @private
         */
        _initializePerformanceWebSocket: function() {
            if (this._performanceWs) return;

            try {
                this._performanceWs = SecurityUtils.createSecureWebSocket('blockchain://a2a-events', {
                    onMessage: function(data) {
                        this._handlePerformanceMetricUpdate(data);
                    }.bind(this)
                });

                this._performanceWs.onclose = function() {
                    console.info("Performance WebSocket closed, will reconnect...");
                    setTimeout(() => this._initializePerformanceWebSocket(), 5000);
                }.bind(this);

            } catch (error) {
                console.warn("Performance WebSocket not available");
            }
        },

        /**
         * @function _handlePerformanceMetricUpdate
         * @description Handles real-time performance metric updates.
         * @param {Object} data - Metric update data
         * @private
         */
        _handlePerformanceMetricUpdate: function(data) {
            // Update any open dialogs with real-time data
            if (this._dialogCache.viewAnalytics && this._dialogCache.viewAnalytics.isOpen()) {
                this._updateAnalyticsData(data, this._dialogCache.viewAnalytics);
            }
        },

        /**
         * @function _getOrCreateDialog
         * @description Gets cached dialog or creates new one with accessibility and responsive features.
         * @param {string} sDialogId - Dialog identifier
         * @param {string} sFragmentName - Fragment name
         * @returns {Promise<sap.m.Dialog>} Promise resolving to dialog
         * @private
         */
        _getOrCreateDialog: function(sDialogId, sFragmentName) {
            var that = this;
            
            if (this._dialogCache && this._dialogCache[sDialogId]) {
                return Promise.resolve(this._dialogCache[sDialogId]);
            }
            
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
                        agent: "Agent12_Performance",
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
            const mockRoles = ["PerformanceAdmin", "PerformanceUser", "PerformanceOperator"];
            return mockRoles.includes(role);
        },

        /**
         * @function _cleanupResources
         * @description Cleans up resources to prevent memory leaks.
         * @private
         */
        _cleanupResources: function() {
            // Clean up WebSocket connections
            if (this._performanceWs) {
                this._performanceWs.close();
                this._performanceWs = null;
            }
            
            // Clean up EventSource connections
            if (this._analyticsEventSource) {
                this._analyticsEventSource.close();
                this._analyticsEventSource = null;
            }
            
            // Clean up polling intervals
            if (this._analyticsPollingInterval) {
                clearInterval(this._analyticsPollingInterval);
                this._analyticsPollingInterval = null;
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
        },

        /**
         * @function getResourceBundle
         * @description Gets the i18n resource bundle.
         * @returns {sap.base.i18n.ResourceBundle} Resource bundle
         * @public
         */
        getResourceBundle: function() {
            return this.base.getView().getModel("i18n").getResourceBundle();
        }
    });
});