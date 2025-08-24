sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel",
    "a2a/network/agent10/ext/utils/SecurityUtils"
], function(ControllerExtension, MessageToast, MessageBox, Fragment, JSONModel, SecurityUtils) {
    "use strict";

    /**
     * @class a2a.network.agent10.ext.controller.ListReportExt
     * @extends sap.ui.core.mvc.ControllerExtension
     * @description Controller extension for Agent 10 List Report - Mathematical Calculation Agent.
     * Provides comprehensive mathematical computation capabilities including formula builders,
     * statistical analysis, precision validation, self-healing algorithms, and performance optimization.
     */
    return ControllerExtension.extend("a2a.network.agent10.ext.controller.ListReportExt", {
        
        override: {
            /**
             * @function onInit
             * @description Initializes the controller extension with security utilities, device model, dialog caching, and real-time updates.
             * @override
             */
            onInit: function () {
                this._extensionAPI = this.base.getExtensionAPI();
                this._securityUtils = SecurityUtils;
                this._initializeDeviceModel();
                this._initializeDialogCache();
                this._initializePerformanceOptimizations();
                this._startRealtimeCalculationUpdates();
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
            // Implement search logic for calculation tasks
        },

        /**
         * @function onStartMonitoring
         * @description Starts monitoring for selected monitoring tasks with real-time updates.
         * @public
         */
        onStartMonitoring: function() {
            if (!this._hasRole("MonitoringAdmin")) {
                MessageBox.error("Access denied. Monitoring Administrator role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "StartMonitoring", reason: "Insufficient permissions" });
                return;
            }

            const oBinding = this.base.getView().byId("fe::table::MonitoringTasks::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectTasksFirst"));
                return;
            }

            this._auditLogger.log("START_MONITORING", { taskCount: aSelectedContexts.length });
            
            MessageBox.confirm(
                this.getResourceBundle().getText("msg.startMonitoringConfirm", [aSelectedContexts.length]),
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._executeStartMonitoring(aSelectedContexts);
                        }
                    }.bind(this)
                }
            );
        },

        /**
         * @function onViewDashboard
         * @description Opens comprehensive monitoring dashboard with real-time metrics and analytics.
         * @public
         */
        onViewDashboard: function() {
            if (!this._hasRole("MonitoringUser")) {
                MessageBox.error("Access denied. Monitoring User role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "ViewDashboard", reason: "Insufficient permissions" });
                return;
            }

            this._auditLogger.log("VIEW_DASHBOARD", { action: "OpenMonitoringDashboard" });
            
            this._getOrCreateDialog("monitoringDashboard", "a2a.network.agent10.ext.fragment.MonitoringDashboard")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadDashboardData(oDialog);
                    this._startRealtimeMetricsStream(oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Monitoring Dashboard: " + error.message);
                });
        },

        /**
         * @function onConfigureAlerts
         * @description Opens alert configuration interface for monitoring thresholds and notifications.
         * @public
         */
        onConfigureAlerts: function() {
            if (!this._hasRole("MonitoringAdmin")) {
                MessageBox.error("Access denied. Monitoring Administrator role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "ConfigureAlerts", reason: "Insufficient permissions" });
                return;
            }

            this._auditLogger.log("CONFIGURE_ALERTS", { action: "OpenAlertConfiguration" });
            
            this._getOrCreateDialog("alertConfiguration", "a2a.network.agent10.ext.fragment.AlertConfiguration")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadAlertConfigurations(oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Alert Configuration: " + error.message);
                });
        },

        /**
         * @function onCalculationDashboard
         * @description Opens comprehensive calculation analytics dashboard with performance metrics.
         * @public
         */
        onCalculationDashboard: function() {
            this._getOrCreateDialog("calculationDashboard", "a2a.network.agent10.ext.fragment.CalculationDashboard")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadDashboardData(oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Calculation Dashboard: " + error.message);
                });
        },

        /**
         * @function onCreateCalculationTask
         * @description Opens dialog to create new mathematical calculation task.
         * @public
         */
        onCreateCalculationTask: function() {
            this._getOrCreateDialog("createCalculationTask", "a2a.network.agent10.ext.fragment.CreateCalculationTask")
                .then(function(oDialog) {
                    var oModel = new JSONModel({
                        taskName: "",
                        description: "",
                        calculationType: "BASIC",
                        formulaExpression: "",
                        precisionLevel: "DOUBLE",
                        calculationEngine: "NUMPY",
                        parallelProcessing: true,
                        gpuAcceleration: false,
                        selfHealingEnabled: true,
                        priority: "MEDIUM"
                    });
                    oDialog.setModel(oModel, "create");
                    oDialog.open();
                    this._loadCalculationOptions(oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Create Task dialog: " + error.message);
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
                        var sRetryMsg = oBundle.getText("recovery.retrying") || "Network error. Retrying...";
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
         * @function onFormulaBuilder
         * @description Opens advanced formula builder interface for mathematical expressions.
         * @public
         */
        onFormulaBuilder: function() {
            this._getOrCreateDialog("formulaBuilder", "a2a.network.agent10.ext.fragment.FormulaBuilder")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadFormulaLibrary(oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Formula Builder: " + error.message);
                });
        },

        /**
         * @function onStatisticalAnalyzer
         * @description Opens statistical analysis interface for data analysis and hypothesis testing.
         * @public
         */
        onStatisticalAnalyzer: function() {
            this._getOrCreateDialog("statisticalAnalyzer", "a2a.network.agent10.ext.fragment.StatisticalAnalysis")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadStatisticalFunctions(oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Statistical Analyzer: " + error.message);
                });
        },

        /**
         * @function onEngineManager
         * @description Opens calculation engine management interface for performance monitoring.
         * @public
         */
        onEngineManager: function() {
            this._getOrCreateDialog("engineManager", "a2a.network.agent10.ext.fragment.EngineManager")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadEngineData(oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Engine Manager: " + error.message);
                });
        },

        /**
         * @function onPrecisionValidator
         * @description Validates precision of selected calculation tasks.
         * @public
         */
        onPrecisionValidator: function() {
            const oBinding = this.base.getView().byId("fe::table::CalculationTasks::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectTasksFirst"));
                return;
            }

            this._runPrecisionValidation(aSelectedContexts);
        },

        /**
         * @function onBatchCalculator
         * @description Executes batch calculation on selected tasks.
         * @public
         */
        onBatchCalculator: function() {
            const oBinding = this.base.getView().byId("fe::table::CalculationTasks::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectTasksFirst"));
                return;
            }

            MessageBox.confirm(
                this.getResourceBundle().getText("msg.batchCalculationConfirm", [aSelectedContexts.length]),
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._runBatchCalculation(aSelectedContexts);
                        }
                    }.bind(this)
                }
            );
        },

        /**
         * @function onPerformanceOptimizer
         * @description Opens performance optimization interface for calculation engines.
         * @public
         */
        onPerformanceOptimizer: function() {
            this._getOrCreateDialog("performanceOptimizer", "a2a.network.agent10.ext.fragment.PerformanceOptimizer")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadPerformanceData(oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Performance Optimizer: " + error.message);
                });
        },

        /**
         * @function _executeStartMonitoring
         * @description Executes monitoring start operation for selected tasks with progress tracking.
         * @param {Array} aSelectedContexts - Selected monitoring task contexts
         * @private
         */
        _executeStartMonitoring: function(aSelectedContexts) {
            const aTaskIds = aSelectedContexts.map(ctx => ctx.getObject().taskId);
            
            // Show progress dialog
            this._getOrCreateDialog("monitoringProgress", "a2a.network.agent10.ext.fragment.MonitoringProgress")
                .then(function(oProgressDialog) {
                    var oProgressModel = new JSONModel({
                        totalTasks: aTaskIds.length,
                        completedTasks: 0,
                        currentTask: "",
                        progress: 0,
                        status: "Starting monitoring tasks..."
                    });
                    oProgressDialog.setModel(oProgressModel, "progress");
                    oProgressDialog.open();
                    
                    this._startMonitoringTasks(aTaskIds, oProgressDialog);
                }.bind(this));
        },

        /**
         * @function _startMonitoringTasks
         * @description Starts monitoring tasks with real-time progress updates.
         * @param {Array} aTaskIds - Array of task IDs to start monitoring
         * @param {sap.m.Dialog} oProgressDialog - Progress dialog
         * @private
         */
        _startMonitoringTasks: function(aTaskIds, oProgressDialog) {
            const oModel = this.base.getView().getModel();
            
            SecurityUtils.secureCallFunction(oModel, "/StartMonitoring", {
                urlParameters: {
                    taskIds: aTaskIds.join(',')
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.monitoringStarted"));
                    oProgressDialog.close();
                    this._refreshMonitoringData();
                    this._auditLogger.log("MONITORING_STARTED", { taskCount: aTaskIds.length, success: true });
                }.bind(this),
                error: function(error) {
                    MessageBox.error(this.getResourceBundle().getText("error.startMonitoringFailed"));
                    oProgressDialog.close();
                    this._auditLogger.log("MONITORING_FAILED", { taskCount: aTaskIds.length, error: error.message });
                }.bind(this)
            });
        },

        /**
         * @function _startRealtimeMetricsStream
         * @description Starts real-time metrics streaming for dashboard.
         * @param {sap.m.Dialog} oDialog - Dashboard dialog
         * @private
         */
        _startRealtimeMetricsStream: function(oDialog) {
            if (this._metricsEventSource) {
                this._metricsEventSource.close();
            }
            
            try {
                this._metricsEventSource = new EventSource('/api/agent10/monitoring/metrics-stream');
                
                this._metricsEventSource.onmessage = function(event) {
                    try {
                        const data = JSON.parse(event.data);
                        this._updateDashboardMetrics(data, oDialog);
                    } catch (error) {
                        console.error('Error parsing metrics data:', error);
                    }
                }.bind(this);
                
                this._metricsEventSource.onerror = function(error) {
                    console.warn('Metrics stream error, falling back to polling:', error);
                    this._startMetricsPolling(oDialog);
                }.bind(this);
                
            } catch (error) {
                console.warn('EventSource not available, using polling fallback');
                this._startMetricsPolling(oDialog);
            }
        },

        /**
         * @function _startMetricsPolling
         * @description Starts polling fallback for metrics updates.
         * @param {sap.m.Dialog} oDialog - Dashboard dialog
         * @private
         */
        _startMetricsPolling: function(oDialog) {
            if (this._metricsPollingInterval) {
                clearInterval(this._metricsPollingInterval);
            }
            
            this._metricsPollingInterval = setInterval(() => {
                this._fetchMetricsUpdate(oDialog);
            }, 5000);
        },

        /**
         * @function _fetchMetricsUpdate
         * @description Fetches metrics update via polling.
         * @param {sap.m.Dialog} oDialog - Dashboard dialog
         * @private
         */
        _fetchMetricsUpdate: function(oDialog) {
            const oModel = this.base.getView().getModel();
            
            SecurityUtils.secureCallFunction(oModel, "/GetRealtimeMetrics", {
                success: function(data) {
                    this._updateDashboardMetrics(data, oDialog);
                }.bind(this),
                error: function(error) {
                    console.warn('Failed to fetch metrics update:', error);
                }
            });
        },

        /**
         * @function _updateDashboardMetrics
         * @description Updates dashboard with real-time metrics.
         * @param {Object} data - Metrics data
         * @param {sap.m.Dialog} oDialog - Dashboard dialog
         * @private
         */
        _updateDashboardMetrics: function(data, oDialog) {
            if (!oDialog || !oDialog.isOpen()) return;
            
            var oDashboardModel = oDialog.getModel("dashboard");
            if (oDashboardModel) {
                var oCurrentData = oDashboardModel.getData();
                oCurrentData.realtimeMetrics = data;
                oCurrentData.lastUpdated = new Date().toISOString();
                oDashboardModel.setData(oCurrentData);
            }
        },

        /**
         * @function _loadAlertConfigurations
         * @description Loads alert configuration data.
         * @param {sap.m.Dialog} oDialog - Alert configuration dialog
         * @private
         */
        _loadAlertConfigurations: function(oDialog) {
            oDialog.setBusy(true);
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    var oModel = this.base.getView().getModel();
                    SecurityUtils.secureCallFunction(oModel, "/GetAlertConfigurations", {
                        success: function(data) {
                            var oAlertModel = new JSONModel({
                                configurations: data.configurations,
                                thresholds: data.thresholds,
                                notificationChannels: data.notificationChannels,
                                alertTypes: data.alertTypes
                            });
                            oDialog.setModel(oAlertModel, "alerts");
                            resolve(data);
                        },
                        error: function(error) {
                            reject(new Error("Failed to load alert configurations"));
                        }
                    });
                }.bind(this));
            }.bind(this)).then(function(data) {
                oDialog.setBusy(false);
            }.bind(this)).catch(function(error) {
                oDialog.setBusy(false);
                MessageBox.error(error.message);
            });
        },

        /**
         * @function _refreshMonitoringData
         * @description Refreshes monitoring task data in the table.
         * @private
         */
        _refreshMonitoringData: function() {
            const oBinding = this.base.getView().byId("fe::table::MonitoringTasks::LineItem").getBinding("rows");
            oBinding.refresh();
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
            const mockRoles = ["MonitoringAdmin", "MonitoringUser", "MonitoringOperator"];
            return mockRoles.includes(role);
        },

        /**
         * @function _startRealtimeCalculationUpdates
         * @description Starts real-time updates for calculation progress and performance metrics.
         * @private
         */
        _startRealtimeCalculationUpdates: function() {
            this._initializeWebSocket();
            this._initializeSecurity();
        },

        /**
         * @function _initializeWebSocket
         * @description Initializes secure WebSocket connection for real-time updates.
         * @private
         */
        _initializeWebSocket: function() {
            if (this._ws) return;

            // Validate WebSocket URL for security
            if (!this._securityUtils.validateWebSocketUrl('ws://localhost:8010/calculations/updates')) {
                MessageBox.error("Invalid WebSocket URL");
                return;
            }

            try {
                this._ws = SecurityUtils.createSecureWebSocket('ws://localhost:8010/calculations/updates', {
                    onMessage: function(data) {
                        this._handleCalculationUpdate(data);
                    }.bind(this)
                });

                this._ws.onclose = function() {
                    var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                    var sMessage = oBundle.getText("msg.websocketDisconnected") || "Connection lost. Reconnecting...";
                    MessageToast.show(sMessage);
                    setTimeout(() => this._initializeWebSocket(), 5000);
                }.bind(this);

            } catch (error) {
                console.warn("WebSocket connection failed, falling back to polling");
                this._initializePolling();
            }
        },

        /**
         * @function _initializePolling
         * @description Initializes polling fallback for real-time updates.
         * @private
         */
        _initializePolling: function() {
            this._pollInterval = setInterval(() => {
                this._refreshCalculationData();
            }, 5000);
        },

        /**
         * @function _handleCalculationUpdate
         * @description Handles real-time calculation updates from WebSocket.
         * @param {Object} data - Update data
         * @private
         */
        _handleCalculationUpdate: function(data) {
            try {
                var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                
                switch (data.type) {
                    case 'CALCULATION_STARTED':
                        var sStartMsg = oBundle.getText("msg.calculationStarted") || "Calculation started";
                        MessageToast.show(sStartMsg);
                        break;
                    case 'CALCULATION_COMPLETED':
                        var sCompleteMsg = oBundle.getText("msg.calculationCompleted") || "Calculation completed";
                        MessageToast.show(sCompleteMsg);
                        this._refreshCalculationData();
                        break;
                    case 'CALCULATION_FAILED':
                        var sErrorMsg = oBundle.getText("error.calculationFailed") || "Calculation failed";
                        var safeError = SecurityUtils.escapeHTML(data.error || 'Unknown error');
                        MessageToast.show(sErrorMsg + ": " + safeError);
                        break;
                    case 'PERFORMANCE_UPDATE':
                        this._updatePerformanceMetrics(data.metrics);
                        break;
                    case 'SELF_HEALING_TRIGGERED':
                        var sHealingMsg = oBundle.getText("msg.selfHealingTriggered") || "Self-healing triggered";
                        MessageToast.show(sHealingMsg);
                        break;
                }
            } catch (error) {
                console.error("Error processing calculation update:", error);
            }
        },

        /**
         * @function _loadDashboardData
         * @description Loads calculation dashboard data with statistics and performance metrics.
         * @param {sap.m.Dialog} oDialog - Target dialog (optional)
         * @private
         */
        _loadDashboardData: function(oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["calculationDashboard"];
            if (!oTargetDialog) return;
            
            oTargetDialog.setBusy(true);
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    var oModel = this.base.getView().getModel();
                    SecurityUtils.secureCallFunction(oModel, "/GetCalculationStatistics", {
                        success: function(data) {
                            resolve(data);
                        },
                        error: function(error) {
                            var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                            var sErrorMsg = oBundle.getText("error.loadingStatistics") || "Error loading statistics";
                            reject(new Error(sErrorMsg));
                        }.bind(this)
                    });
                }.bind(this));
            }.bind(this)).then(function(data) {
                oTargetDialog.setBusy(false);
                this._updateDashboardCharts(data, oTargetDialog);
            }.bind(this)).catch(function(error) {
                oTargetDialog.setBusy(false);
                MessageBox.error(error.message);
            });
        },

        /**
         * @function _loadEngineData
         * @description Loads calculation engine data and status information.
         * @param {sap.m.Dialog} oDialog - Target dialog (optional)
         * @private
         */
        _loadEngineData: function(oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["engineManager"];
            if (!oTargetDialog) return;
            
            oTargetDialog.setBusy(true);
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    var oModel = this.base.getView().getModel();
                    SecurityUtils.secureCallFunction(oModel, "/GetEngineStatus", {
                        success: function(data) {
                            resolve(data);
                        },
                        error: function(error) {
                            var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                            var sErrorMsg = oBundle.getText("error.loadingEngineStatus") || "Error loading engine status";
                            reject(new Error(sErrorMsg));
                        }.bind(this)
                    });
                }.bind(this));
            }.bind(this)).then(function(data) {
                oTargetDialog.setBusy(false);
                this._updateEngineStatus(data, oTargetDialog);
            }.bind(this)).catch(function(error) {
                oTargetDialog.setBusy(false);
                MessageBox.error(error.message);
            });
        },

        /**
         * @function _loadPerformanceData
         * @description Loads performance metrics and optimization data.
         * @param {sap.m.Dialog} oDialog - Target dialog (optional)
         * @private
         */
        _loadPerformanceData: function(oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["performanceOptimizer"];
            if (!oTargetDialog) return;
            
            oTargetDialog.setBusy(true);
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    var oModel = this.base.getView().getModel();
                    SecurityUtils.secureCallFunction(oModel, "/GetPerformanceMetrics", {
                        success: function(data) {
                            resolve(data);
                        },
                        error: function(error) {
                            var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                            var sErrorMsg = oBundle.getText("error.loadingPerformanceData") || "Error loading performance data";
                            reject(new Error(sErrorMsg));
                        }.bind(this)
                    });
                }.bind(this));
            }.bind(this)).then(function(data) {
                oTargetDialog.setBusy(false);
                this._updatePerformanceCharts(data, oTargetDialog);
            }.bind(this)).catch(function(error) {
                oTargetDialog.setBusy(false);
                MessageBox.error(error.message);
            });
        },

        _runPrecisionValidation: function(aSelectedContexts) {
            const oModel = this.getView().getModel();
            const aTaskIds = aSelectedContexts.map(ctx => ctx.getObject().taskId);
            
            SecurityUtils.secureCallFunction(oModel, "/ValidatePrecision", {
                urlParameters: {
                    taskIds: aTaskIds.join(',')
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.precisionVerified"));
                    this._refreshCalculationData();
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.precisionValidationFailed"));
                }.bind(this)
            });
        },

        _runBatchCalculation: function(aSelectedContexts) {
            const oModel = this.getView().getModel();
            const aTaskIds = aSelectedContexts.map(ctx => ctx.getObject().taskId);
            
            MessageToast.show(this.getResourceBundle().getText("msg.batchCalculationStarted", [aTaskIds.length]));
            
            SecurityUtils.secureCallFunction(oModel, "/ExecuteBatchCalculation", {
                urlParameters: {
                    taskIds: aTaskIds.join(',')
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.batchCalculationCompleted"));
                    this._refreshCalculationData();
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.batchCalculationFailed"));
                }.bind(this)
            });
        },

        _refreshCalculationData: function() {
            const oBinding = this.base.getView().byId("fe::table::CalculationTasks::LineItem").getBinding("rows");
            oBinding.refresh();
        },

        /**
         * @function _loadCalculationOptions
         * @description Loads calculation options and engine configurations.
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _loadCalculationOptions: function(oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["createCalculationTask"];
            if (!oTargetDialog) return;
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    var oModel = this.base.getView().getModel();
                    SecurityUtils.secureCallFunction(oModel, "/GetCalculationOptions", {
                        success: function(data) {
                            var oCreateModel = oTargetDialog.getModel("create");
                            var oCreateData = oCreateModel.getData();
                            oCreateData.availableEngines = data.engines;
                            oCreateData.formulaCategories = data.categories;
                            oCreateData.precisionLevels = data.precisionLevels;
                            oCreateModel.setData(oCreateData);
                            resolve(data);
                        },
                        error: function(error) {
                            reject(new Error("Failed to load calculation options"));
                        }
                    });
                }.bind(this));
            }.bind(this)).catch(function(error) {
                MessageBox.error(error.message);
            });
        },
        
        /**
         * @function _loadFormulaLibrary
         * @description Loads formula library and mathematical functions.
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _loadFormulaLibrary: function(oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["formulaBuilder"];
            if (!oTargetDialog) return;
            
            oTargetDialog.setBusy(true);
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    var oModel = this.base.getView().getModel();
                    SecurityUtils.secureCallFunction(oModel, "/GetFormulaLibrary", {
                        success: function(data) {
                            var oFormulaModel = new JSONModel({
                                functions: data.functions,
                                operators: data.operators,
                                constants: data.constants,
                                examples: data.examples
                            });
                            oTargetDialog.setModel(oFormulaModel, "formula");
                            resolve(data);
                        },
                        error: function(error) {
                            reject(new Error("Failed to load formula library"));
                        }
                    });
                }.bind(this));
            }.bind(this)).then(function(data) {
                oTargetDialog.setBusy(false);
            }.bind(this)).catch(function(error) {
                oTargetDialog.setBusy(false);
                MessageBox.error(error.message);
            });
        },
        
        /**
         * @function _loadStatisticalFunctions
         * @description Loads statistical analysis functions and configurations.
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _loadStatisticalFunctions: function(oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["statisticalAnalyzer"];
            if (!oTargetDialog) return;
            
            oTargetDialog.setBusy(true);
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    var oModel = this.base.getView().getModel();
                    SecurityUtils.secureCallFunction(oModel, "/GetStatisticalFunctions", {
                        success: function(data) {
                            var oStatModel = new JSONModel({
                                functions: data.functions,
                                testTypes: data.testTypes,
                                confidenceLevels: data.confidenceLevels,
                                datasets: data.availableDatasets
                            });
                            oTargetDialog.setModel(oStatModel, "statistical");
                            resolve(data);
                        },
                        error: function(error) {
                            reject(new Error("Failed to load statistical functions"));
                        }
                    });
                }.bind(this));
            }.bind(this)).then(function(data) {
                oTargetDialog.setBusy(false);
            }.bind(this)).catch(function(error) {
                oTargetDialog.setBusy(false);
                MessageBox.error(error.message);
            });
        },

        /**
         * @function _updateDashboardCharts
         * @description Updates calculation dashboard charts with performance metrics.
         * @param {Object} data - Dashboard data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _updateDashboardCharts: function(data, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["calculationDashboard"];
            if (!oTargetDialog) return;
            
            this._createPerformanceTrendsChart(data.performanceTrends, oTargetDialog);
            this._createAccuracyMetricsChart(data.accuracyMetrics, oTargetDialog);
            this._createEngineComparisonChart(data.engineComparison, oTargetDialog);
        },

        /**
         * @function _updateEngineStatus
         * @description Updates calculation engine status and resource utilization.
         * @param {Object} data - Engine status data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _updateEngineStatus: function(data, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["engineManager"];
            if (!oTargetDialog) return;
            
            var oEngineModel = new JSONModel({
                engines: data.engines,
                resourceUtilization: data.resourceUtilization,
                performance: data.performance,
                healthStatus: data.healthStatus
            });
            oTargetDialog.setModel(oEngineModel, "engines");
            
            this._createEngineStatusVisualizations(data, oTargetDialog);
        },

        /**
         * @function _updatePerformanceCharts
         * @description Updates performance optimization charts and metrics.
         * @param {Object} data - Performance data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _updatePerformanceCharts: function(data, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["performanceOptimizer"];
            if (!oTargetDialog) return;
            
            var oPerformanceModel = new JSONModel({
                throughputAnalysis: data.throughputAnalysis,
                resourceEfficiency: data.resourceEfficiency,
                scalabilityMetrics: data.scalabilityMetrics,
                optimizationSuggestions: data.optimizationSuggestions
            });
            oTargetDialog.setModel(oPerformanceModel, "performance");
            
            this._createThroughputAnalysisChart(data.throughputAnalysis, oTargetDialog);
            this._createResourceEfficiencyChart(data.resourceEfficiency, oTargetDialog);
            this._createScalabilityMetricsChart(data.scalabilityMetrics, oTargetDialog);
        },

        /**
         * @function _updatePerformanceMetrics
         * @description Updates real-time performance metrics from WebSocket.
         * @param {Object} metrics - Performance metrics
         * @private
         */
        _updatePerformanceMetrics: function(metrics) {
            // Update dashboard if open
            var oDashboard = this._dialogCache["calculationDashboard"];
            if (oDashboard && oDashboard.isOpen()) {
                var oDashboardModel = oDashboard.getModel("dashboard");
                if (oDashboardModel) {
                    var oData = oDashboardModel.getData();
                    oData.realtimeMetrics = metrics;
                    oDashboardModel.setData(oData);
                }
            }
            
            // Update performance optimizer if open
            var oOptimizer = this._dialogCache["performanceOptimizer"];
            if (oOptimizer && oOptimizer.isOpen()) {
                var oOptimizerModel = oOptimizer.getModel("performance");
                if (oOptimizerModel) {
                    var oData = oOptimizerModel.getData();
                    oData.realtimeMetrics = metrics;
                    oOptimizerModel.setData(oData);
                }
            }
        },

        /**
         * @function _createPerformanceTrendsChart
         * @description Creates performance trends visualization chart.
         * @param {Object} trendsData - Performance trends data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createPerformanceTrendsChart: function(trendsData, oDialog) {
            if (!trendsData || !oDialog) return;
            
            var oChart = oDialog.byId("performanceTrendsChart");
            if (!oChart) return;
            
            var aChartData = trendsData.map(function(trend) {
                return {
                    Time: trend.timestamp,
                    Throughput: trend.throughput,
                    Accuracy: trend.accuracy,
                    ResponseTime: trend.responseTime
                };
            });
            
            var oChartModel = new JSONModel({ trendsData: aChartData });
            oChart.setModel(oChartModel);
        },
        
        /**
         * @function _createAccuracyMetricsChart
         * @description Creates accuracy metrics visualization chart.
         * @param {Object} accuracyData - Accuracy metrics data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createAccuracyMetricsChart: function(accuracyData, oDialog) {
            if (!accuracyData || !oDialog) return;
            
            var oChart = oDialog.byId("accuracyMetricsChart");
            if (!oChart) return;
            
            var aChartData = accuracyData.map(function(metric) {
                return {
                    Engine: metric.engineName,
                    Accuracy: metric.accuracy,
                    Precision: metric.precision,
                    ErrorRate: metric.errorRate
                };
            });
            
            var oChartModel = new JSONModel({ accuracyData: aChartData });
            oChart.setModel(oChartModel);
        },
        
        /**
         * @function _createEngineComparisonChart
         * @description Creates calculation engine comparison chart.
         * @param {Object} comparisonData - Engine comparison data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createEngineComparisonChart: function(comparisonData, oDialog) {
            if (!comparisonData || !oDialog) return;
            
            var oChart = oDialog.byId("engineComparisonChart");
            if (!oChart) return;
            
            var aChartData = comparisonData.map(function(engine) {
                return {
                    Engine: engine.engineName,
                    Performance: engine.performanceScore,
                    Reliability: engine.reliabilityScore,
                    Efficiency: engine.efficiencyScore
                };
            });
            
            var oChartModel = new JSONModel({ comparisonData: aChartData });
            oChart.setModel(oChartModel);
        },
        
        /**
         * @function _createEngineStatusVisualizations
         * @description Creates engine status and health visualizations.
         * @param {Object} data - Engine status data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createEngineStatusVisualizations: function(data, oDialog) {
            if (!data || !oDialog) return;
            
            var oHealthChart = oDialog.byId("engineHealthChart");
            if (oHealthChart && data.healthStatus) {
                var aHealthData = data.healthStatus.map(function(status) {
                    return {
                        Engine: status.engineName,
                        Health: status.healthScore,
                        Uptime: status.uptime,
                        Load: status.currentLoad
                    };
                });
                
                var oHealthModel = new JSONModel({ healthData: aHealthData });
                oHealthChart.setModel(oHealthModel);
            }
        },
        
        /**
         * @function _createThroughputAnalysisChart
         * @description Creates throughput analysis visualization.
         * @param {Object} throughputData - Throughput analysis data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createThroughputAnalysisChart: function(throughputData, oDialog) {
            if (!throughputData || !oDialog) return;
            
            var oChart = oDialog.byId("throughputAnalysisChart");
            if (!oChart) return;
            
            var aChartData = throughputData.map(function(data) {
                return {
                    Time: data.timestamp,
                    Operations: data.operationsPerSecond,
                    Requests: data.requestsPerSecond,
                    Calculations: data.calculationsPerSecond
                };
            });
            
            var oChartModel = new JSONModel({ throughputData: aChartData });
            oChart.setModel(oChartModel);
        },
        
        /**
         * @function _createResourceEfficiencyChart
         * @description Creates resource efficiency visualization.
         * @param {Object} efficiencyData - Resource efficiency data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createResourceEfficiencyChart: function(efficiencyData, oDialog) {
            if (!efficiencyData || !oDialog) return;
            
            var oChart = oDialog.byId("resourceEfficiencyChart");
            if (!oChart) return;
            
            var aChartData = efficiencyData.map(function(data) {
                return {
                    Resource: data.resourceType,
                    Utilization: data.utilization,
                    Efficiency: data.efficiency,
                    Cost: data.cost
                };
            });
            
            var oChartModel = new JSONModel({ efficiencyData: aChartData });
            oChart.setModel(oChartModel);
        },
        
        /**
         * @function _createScalabilityMetricsChart
         * @description Creates scalability metrics visualization.
         * @param {Object} scalabilityData - Scalability metrics data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createScalabilityMetricsChart: function(scalabilityData, oDialog) {
            if (!scalabilityData || !oDialog) return;
            
            var oChart = oDialog.byId("scalabilityMetricsChart");
            if (!oChart) return;
            
            var aChartData = scalabilityData.map(function(data) {
                return {
                    Load: data.loadLevel,
                    ResponseTime: data.responseTime,
                    Throughput: data.throughput,
                    ErrorRate: data.errorRate
                };
            });
            
            var oChartModel = new JSONModel({ scalabilityData: aChartData });
            oChart.setModel(oChartModel);
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
                        agent: "Agent10_Monitoring",
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
         * @function _cleanupResources
         * @description Cleans up resources to prevent memory leaks.
         * @private
         */
        _cleanupResources: function() {
            // Clean up WebSocket connections
            if (this._ws) {
                this._ws.close();
                this._ws = null;
            }
            
            // Clean up EventSource connections
            if (this._metricsEventSource) {
                this._metricsEventSource.close();
                this._metricsEventSource = null;
            }
            
            // Clean up polling intervals
            if (this._pollInterval) {
                clearInterval(this._pollInterval);
                this._pollInterval = null;
            }
            
            if (this._metricsPollingInterval) {
                clearInterval(this._metricsPollingInterval);
                this._metricsPollingInterval = null;
            }
            
            // Clean up cached dialogs
            Object.keys(this._dialogCache).forEach(function(key) {
                if (this._dialogCache[key]) {
                    this._dialogCache[key].destroy();
                }
            }.bind(this));
            this._dialogCache = {};
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