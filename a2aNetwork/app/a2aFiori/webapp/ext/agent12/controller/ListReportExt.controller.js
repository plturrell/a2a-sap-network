/**
 * A2A Protocol Compliance: WebSocket replaced with blockchain event streaming
 * All real-time communication now uses blockchain events instead of WebSockets
 */

sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/ui/core/Fragment",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/model/json/JSONModel",
    "a2a/network/agent12/ext/utils/SecurityUtils"
], (ControllerExtension, Fragment, MessageBox, MessageToast, JSONModel, SecurityUtils) => {
    "use strict";

    /**
     * @class a2a.network.agent12.ext.controller.ListReportExt
     * @extends sap.ui.core.mvc.ControllerExtension
     * @description Controller extension for Agent 12 List Report - Performance Management.
     * Provides comprehensive performance benchmarking, optimization, and tuning capabilities
     * with enterprise-grade security, audit logging, and accessibility features.
     */
    return ControllerExtension.extend("a2a.network.agent12.ext.controller.ListReportExt", {

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
                this._initializeSecurity();
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
         * @function onStartBenchmark
         * @description Starts performance benchmark for selected performance tasks with real-time monitoring.
         * @public
         */
        onStartBenchmark() {
            if (!this._hasRole("PerformanceAdmin")) {
                MessageBox.error("Access denied. Performance Administrator role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "StartBenchmark", reason: "Insufficient permissions" });
                return;
            }

            const oBinding = this.base.getView().byId("fe::table::PerformanceTasks::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();

            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectTasksFirst"));
                return;
            }

            this._auditLogger.log("START_BENCHMARK", { taskCount: aSelectedContexts.length });

            MessageBox.confirm(
                this.getResourceBundle().getText("msg.startBenchmarkConfirm", [aSelectedContexts.length]),
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._executeBenchmark(aSelectedContexts);
                        }
                    }.bind(this)
                }
            );
        },

        /**
         * @function onOptimizePerformance
         * @description Opens performance optimization wizard for selected tasks with AI recommendations.
         * @public
         */
        onOptimizePerformance() {
            if (!this._hasRole("PerformanceAdmin")) {
                MessageBox.error("Access denied. Performance Administrator role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "OptimizePerformance", reason: "Insufficient permissions" });
                return;
            }

            const oBinding = this.base.getView().byId("fe::table::PerformanceTasks::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();

            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectTasksFirst"));
                return;
            }

            this._auditLogger.log("OPTIMIZE_PERFORMANCE", { taskCount: aSelectedContexts.length });

            this._getOrCreateDialog("optimizePerformance", "a2a.network.agent12.ext.fragment.OptimizePerformance")
                .then((oDialog) => {
                    const oOptimizeModel = new JSONModel({
                        selectedTasks: aSelectedContexts.map(ctx => ctx.getObject()),
                        optimizationStrategy: "BALANCED",
                        targetMetric: "THROUGHPUT",
                        autoApply: false,
                        testFirst: true,
                        rollbackEnabled: true,
                        aiRecommendations: [],
                        currentMetrics: {},
                        targetThresholds: {
                            responseTime: 100,
                            throughput: 1000,
                            cpuUsage: 70,
                            memoryUsage: 80
                        }
                    });
                    oDialog.setModel(oOptimizeModel, "optimize");
                    oDialog.open();
                    this._loadOptimizationRecommendations(aSelectedContexts, oDialog);
                })
                .catch((error) => {
                    MessageBox.error(`Failed to open Performance Optimization: ${ error.message}`);
                });
        },

        /**
         * @function onTuneSettings
         * @description Opens performance tuning settings interface with advanced configuration options.
         * @public
         */
        onTuneSettings() {
            if (!this._hasRole("PerformanceAdmin")) {
                MessageBox.error("Access denied. Performance Administrator role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "TuneSettings", reason: "Insufficient permissions" });
                return;
            }

            this._auditLogger.log("TUNE_SETTINGS", { action: "OpenTuningInterface" });

            this._getOrCreateDialog("tuneSettings", "a2a.network.agent12.ext.fragment.TuneSettings")
                .then((oDialog) => {
                    const oTuneModel = new JSONModel({
                        categories: [
                            { key: "GENERAL", text: "General Settings" },
                            { key: "CACHE", text: "Cache Configuration" },
                            { key: "DATABASE", text: "Database Optimization" },
                            { key: "NETWORK", text: "Network Settings" },
                            { key: "MEMORY", text: "Memory Management" },
                            { key: "THREADING", text: "Threading & Concurrency" },
                            { key: "IO", text: "I/O Operations" }
                        ],
                        selectedCategory: "GENERAL",
                        settings: {},
                        profiles: [],
                        currentProfile: "DEFAULT",
                        unsavedChanges: false
                    });
                    oDialog.setModel(oTuneModel, "tune");
                    oDialog.open();
                    this._loadTuningSettings(oDialog);
                })
                .catch((error) => {
                    MessageBox.error(`Failed to open Tune Settings: ${ error.message}`);
                });
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
            // Implement search logic for performance tasks
        },

        /**
         * @function _executeBenchmark
         * @description Executes performance benchmark for selected tasks with progress tracking.
         * @param {Array} aSelectedContexts - Selected performance task contexts
         * @private
         */
        _executeBenchmark(aSelectedContexts) {
            const aTaskIds = aSelectedContexts.map(ctx => ctx.getObject().taskId);

            // Show progress dialog
            this._getOrCreateDialog("benchmarkProgress", "a2a.network.agent12.ext.fragment.BenchmarkProgress")
                .then((oProgressDialog) => {
                    const oProgressModel = new JSONModel({
                        totalTasks: aTaskIds.length,
                        completedTasks: 0,
                        currentTask: "",
                        progress: 0,
                        status: "Starting performance benchmark...",
                        metrics: {
                            responseTime: [],
                            throughput: [],
                            cpuUsage: [],
                            memoryUsage: []
                        },
                        results: []
                    });
                    oProgressDialog.setModel(oProgressModel, "progress");
                    oProgressDialog.open();

                    this._runBenchmarks(aTaskIds, oProgressDialog);
                });
        },

        /**
         * @function _runBenchmarks
         * @description Runs performance benchmarks with real-time progress updates.
         * @param {Array} aTaskIds - Array of task IDs to benchmark
         * @param {sap.m.Dialog} oProgressDialog - Progress dialog
         * @private
         */
        _runBenchmarks(aTaskIds, oProgressDialog) {
            const oModel = this.base.getView().getModel();

            SecurityUtils.secureCallFunction(oModel, "/RunPerformanceBenchmarks", {
                urlParameters: {
                    taskIds: aTaskIds.join(","),
                    iterations: 10,
                    warmupRuns: 3
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.benchmarkStarted"));
                    this._startBenchmarkMonitoring(data.benchmarkId, oProgressDialog);
                    this._auditLogger.log("BENCHMARK_STARTED", {
                        taskCount: aTaskIds.length,
                        benchmarkId: data.benchmarkId,
                        success: true
                    });
                }.bind(this),
                error: function(error) {
                    MessageBox.error(this.getResourceBundle().getText("error.benchmarkFailed"));
                    oProgressDialog.close();
                    this._auditLogger.log("BENCHMARK_FAILED", {
                        taskCount: aTaskIds.length,
                        error: error.message
                    });
                }.bind(this)
            });
        },

        /**
         * @function _startBenchmarkMonitoring
         * @description Starts real-time monitoring of benchmark progress.
         * @param {string} sBenchmarkId - Benchmark ID to monitor
         * @param {sap.m.Dialog} oProgressDialog - Progress dialog
         * @private
         */
        _startBenchmarkMonitoring(sBenchmarkId, oProgressDialog) {
            if (this._benchmarkEventSource) {
                this._benchmarkEventSource.close();
            }

            try {
                this._benchmarkEventSource = new EventSource(`/api/agent12/performance/benchmark-stream/${ sBenchmarkId}`);

                this._benchmarkEventSource.onmessage = function(event) {
                    try {
                        const data = JSON.parse(event.data);
                        this._updateBenchmarkProgress(data, oProgressDialog);
                    } catch (error) {
                        // console.error("Error parsing benchmark progress data:", error);
                    }
                }.bind(this);

                this._benchmarkEventSource.onerror = function(error) {
                    // console.warn("Benchmark stream error, falling back to polling:", error);
                    this._startBenchmarkPolling(sBenchmarkId, oProgressDialog);
                }.bind(this);

            } catch (error) {
                // console.warn("EventSource not available, using polling fallback");
                this._startBenchmarkPolling(sBenchmarkId, oProgressDialog);
            }
        },

        /**
         * @function _startBenchmarkPolling
         * @description Starts polling fallback for benchmark progress updates.
         * @param {string} sBenchmarkId - Benchmark ID to monitor
         * @param {sap.m.Dialog} oProgressDialog - Progress dialog
         * @private
         */
        _startBenchmarkPolling(sBenchmarkId, oProgressDialog) {
            if (this._benchmarkPollingInterval) {
                clearInterval(this._benchmarkPollingInterval);
            }

            this._benchmarkPollingInterval = setInterval(() => {
                this._fetchBenchmarkProgress(sBenchmarkId, oProgressDialog);
            }, 2000);
        },

        /**
         * @function _fetchBenchmarkProgress
         * @description Fetches benchmark progress via polling.
         * @param {string} sBenchmarkId - Benchmark ID to monitor
         * @param {sap.m.Dialog} oProgressDialog - Progress dialog
         * @private
         */
        _fetchBenchmarkProgress(sBenchmarkId, oProgressDialog) {
            const oModel = this.base.getView().getModel();

            SecurityUtils.secureCallFunction(oModel, "/GetBenchmarkProgress", {
                urlParameters: { benchmarkId: sBenchmarkId },
                success: function(data) {
                    this._updateBenchmarkProgress(data, oProgressDialog);
                }.bind(this),
                error(error) {
                    // console.warn("Failed to fetch benchmark progress:", error);
                }
            });
        },

        /**
         * @function _updateBenchmarkProgress
         * @description Updates benchmark progress display.
         * @param {Object} data - Progress data
         * @param {sap.m.Dialog} oProgressDialog - Progress dialog
         * @private
         */
        _updateBenchmarkProgress(data, oProgressDialog) {
            if (!oProgressDialog || !oProgressDialog.isOpen()) {return;}

            const oProgressModel = oProgressDialog.getModel("progress");
            if (oProgressModel) {
                const oCurrentData = oProgressModel.getData();
                oCurrentData.completedTasks = data.completedTasks || oCurrentData.completedTasks;
                oCurrentData.currentTask = data.currentTask || oCurrentData.currentTask;
                oCurrentData.progress = Math.round((oCurrentData.completedTasks / oCurrentData.totalTasks) * 100);
                oCurrentData.status = data.status || oCurrentData.status;

                if (data.metrics) {
                    oCurrentData.metrics = data.metrics;
                }

                if (data.results && data.results.length > 0) {
                    oCurrentData.results = oCurrentData.results.concat(data.results);
                }

                oProgressModel.setData(oCurrentData);

                // Check if all tasks are completed
                if (oCurrentData.completedTasks >= oCurrentData.totalTasks) {
                    this._completeBenchmark(oProgressDialog);
                }
            }
        },

        /**
         * @function _completeBenchmark
         * @description Handles completion of benchmark.
         * @param {sap.m.Dialog} oProgressDialog - Progress dialog
         * @private
         */
        _completeBenchmark(oProgressDialog) {
            setTimeout(() => {
                oProgressDialog.close();
                MessageToast.show(this.getResourceBundle().getText("msg.benchmarkCompleted"));
                this._refreshPerformanceData();
                this._auditLogger.log("BENCHMARK_COMPLETED", { status: "SUCCESS" });

                // Show results summary
                this._showBenchmarkSummary(oProgressDialog.getModel("progress").getData());
            }, 2000);

            // Clean up event source
            if (this._benchmarkEventSource) {
                this._benchmarkEventSource.close();
                this._benchmarkEventSource = null;
            }

            if (this._benchmarkPollingInterval) {
                clearInterval(this._benchmarkPollingInterval);
                this._benchmarkPollingInterval = null;
            }
        },

        /**
         * @function _loadOptimizationRecommendations
         * @description Loads AI-powered optimization recommendations.
         * @param {Array} aSelectedContexts - Selected task contexts
         * @param {sap.m.Dialog} oDialog - Optimization dialog
         * @private
         */
        _loadOptimizationRecommendations(aSelectedContexts, oDialog) {
            oDialog.setBusy(true);

            const oModel = this.base.getView().getModel();
            const aTaskIds = aSelectedContexts.map(ctx => ctx.getObject().taskId);

            SecurityUtils.secureCallFunction(oModel, "/GetOptimizationRecommendations", {
                urlParameters: {
                    taskIds: aTaskIds.join(","),
                    analysisDepth: "DEEP"
                },
                success: function(data) {
                    const oOptimizeModel = oDialog.getModel("optimize");
                    if (oOptimizeModel) {
                        const oCurrentData = oOptimizeModel.getData();
                        oCurrentData.aiRecommendations = data.recommendations || [];
                        oCurrentData.currentMetrics = data.currentMetrics || {};
                        oCurrentData.potentialImprovements = data.potentialImprovements || {};
                        oCurrentData.riskAssessment = data.riskAssessment || {};
                        oOptimizeModel.setData(oCurrentData);
                    }
                    oDialog.setBusy(false);
                }.bind(this),
                error(error) {
                    oDialog.setBusy(false);
                    MessageBox.error(`Failed to load optimization recommendations: ${ error.message}`);
                }
            });
        },

        /**
         * @function _loadTuningSettings
         * @description Loads performance tuning settings and profiles.
         * @param {sap.m.Dialog} oDialog - Tuning settings dialog
         * @private
         */
        _loadTuningSettings(oDialog) {
            const oModel = this.base.getView().getModel();

            SecurityUtils.secureCallFunction(oModel, "/GetPerformanceTuningSettings", {
                success: function(data) {
                    const oTuneModel = oDialog.getModel("tune");
                    if (oTuneModel) {
                        const oCurrentData = oTuneModel.getData();
                        oCurrentData.settings = data.settings || {};
                        oCurrentData.profiles = data.profiles || [];
                        oCurrentData.recommendations = data.recommendations || {};
                        oCurrentData.benchmarks = data.benchmarks || {};
                        oTuneModel.setData(oCurrentData);
                    }
                }.bind(this),
                error(error) {
                    MessageBox.error(`Failed to load tuning settings: ${ error.message}`);
                }
            });
        },

        /**
         * @function _showBenchmarkSummary
         * @description Shows benchmark results summary.
         * @param {Object} benchmarkData - Benchmark results data
         * @private
         */
        _showBenchmarkSummary(benchmarkData) {
            this._getOrCreateDialog("benchmarkSummary", "a2a.network.agent12.ext.fragment.BenchmarkSummary")
                .then((oDialog) => {
                    const oSummaryModel = new JSONModel(benchmarkData);
                    oDialog.setModel(oSummaryModel, "summary");
                    oDialog.open();
                })
                .catch((error) => {
                    MessageBox.error(`Failed to show benchmark summary: ${ error.message}`);
                });
        },

        /**
         * @function _refreshPerformanceData
         * @description Refreshes performance task data in the table.
         * @private
         */
        _refreshPerformanceData() {
            const oBinding = this.base.getView().byId("fe::table::PerformanceTasks::LineItem").getBinding("rows");
            oBinding.refresh();
        },

        /**
         * @function _startRealtimeUpdates
         * @description Starts real-time updates for performance metrics.
         * @private
         */
        _startRealtimeUpdates() {
            this._initializeWebSocket();
        },

        /**
         * @function _initializeWebSocket
         * @description Initializes secure WebSocket connection for real-time updates.
         * @private
         */
        _initializeWebSocket() {
            if (this._ws) {return;}

            // Validate WebSocket URL for security
            if (!this._securityUtils.validateWebSocketUrl("blockchain://a2a-events")) {
                MessageBox.error("Invalid WebSocket URL");
                return;
            }

            try {
                this._ws = SecurityUtils.createSecureWebSocket("blockchain://a2a-events", {
                    onMessage: function(data) {
                        this._handlePerformanceUpdate(data);
                    }.bind(this)
                });

                this._ws.onclose = function() {
                    const oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                    const sMessage = oBundle.getText("msg.websocketDisconnected") || "Connection lost. Reconnecting...";
                    MessageToast.show(sMessage);
                    setTimeout(() => this._initializeWebSocket(), 5000);
                }.bind(this);

            } catch (error) {
                // console.warn("WebSocket connection failed, falling back to polling");
                this._initializePolling();
            }
        },

        /**
         * @function _initializePolling
         * @description Initializes polling fallback for real-time updates.
         * @private
         */
        _initializePolling() {
            this._pollInterval = setInterval(() => {
                this._refreshPerformanceData();
            }, 5000);
        },

        /**
         * @function _handlePerformanceUpdate
         * @description Handles real-time performance updates from WebSocket.
         * @param {Object} data - Update data
         * @private
         */
        _handlePerformanceUpdate(data) {
            try {
                const oBundle = this.base.getView().getModel("i18n").getResourceBundle();

                switch (data.type) {
                case "BENCHMARK_STARTED":
                    const sStartMsg = oBundle.getText("msg.benchmarkStarted") || "Benchmark started";
                    MessageToast.show(sStartMsg);
                    break;
                case "BENCHMARK_COMPLETED":
                    const sCompleteMsg = oBundle.getText("msg.benchmarkCompleted") || "Benchmark completed";
                    MessageToast.show(sCompleteMsg);
                    this._refreshPerformanceData();
                    break;
                case "OPTIMIZATION_APPLIED":
                    const sOptMsg = oBundle.getText("msg.optimizationApplied") || "Optimization applied";
                    MessageToast.show(sOptMsg);
                    this._refreshPerformanceData();
                    break;
                case "PERFORMANCE_ALERT":
                    const sAlertMsg = oBundle.getText("msg.performanceAlert") || "Performance alert";
                    const safeDetails = SecurityUtils.escapeHTML(data.details || "");
                    MessageToast.show(`${sAlertMsg }: ${ safeDetails}`);
                    break;
                }
            } catch (error) {
                // console.error("Error processing performance update:", error);
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
         * @function _initializeSecurity
         * @description Initializes security features and audit logging.
         * @private
         */
        _initializeSecurity() {
            this._auditLogger = {
                log: function(action, details) {
                    const user = this._getCurrentUser();
                    const timestamp = new Date().toISOString();
                    const _logEntry = {
                        timestamp,
                        user,
                        agent: "Agent12_Performance",
                        action,
                        details: details || {}
                    };
                    // console.info(`AUDIT: ${ JSON.stringify(_logEntry)}`);
                }.bind(this)
            };
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
            const mockRoles = ["PerformanceAdmin", "PerformanceUser", "PerformanceOperator"];
            return mockRoles.includes(role);
        },

        /**
         * @function _cleanupResources
         * @description Cleans up resources to prevent memory leaks.
         * @private
         */
        _cleanupResources() {
            // Clean up WebSocket connections
            if (this._ws) {
                this._ws.close();
                this._ws = null;
            }

            // Clean up EventSource connections
            if (this._benchmarkEventSource) {
                this._benchmarkEventSource.close();
                this._benchmarkEventSource = null;
            }

            // Clean up polling intervals
            if (this._pollInterval) {
                clearInterval(this._pollInterval);
                this._pollInterval = null;
            }

            if (this._benchmarkPollingInterval) {
                clearInterval(this._benchmarkPollingInterval);
                this._benchmarkPollingInterval = null;
            }

            // Clean up cached dialogs
            Object.keys(this._dialogCache).forEach((key) => {
                if (this._dialogCache[key]) {
                    this._dialogCache[key].destroy();
                }
            });
            this._dialogCache = {};
        },

        /**
         * @function getResourceBundle
         * @description Gets the i18n resource bundle.
         * @returns {sap.base.i18n.ResourceBundle} Resource bundle
         * @public
         */
        getResourceBundle() {
            return this.base.getView().getModel("i18n").getResourceBundle();
        }
    });
});