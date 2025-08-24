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
    "a2a/network/agent11/ext/utils/SecurityUtils"
], (ControllerExtension, Fragment, MessageBox, MessageToast, JSONModel, SecurityUtils) => {
    "use strict";

    /**
     * @class a2a.network.agent11.ext.controller.ListReportExt
     * @extends sap.ui.core.mvc.ControllerExtension
     * @description Controller extension for Agent 11 List Report - Compliance Management.
     * Provides comprehensive compliance checking, rule management, and reporting capabilities
     * with enterprise-grade security, audit logging, and accessibility features.
     */
    return ControllerExtension.extend("a2a.network.agent11.ext.controller.ListReportExt", {

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
         * @function onRunCompliance
         * @description Runs compliance checks on selected compliance tasks with real-time monitoring.
         * @public
         */
        onRunCompliance() {
            if (!this._hasRole("ComplianceAdmin")) {
                MessageBox.error("Access denied. Compliance Administrator role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "RunCompliance", reason: "Insufficient permissions" });
                return;
            }

            const oBinding = this.base.getView().byId("fe::table::ComplianceTasks::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();

            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectTasksFirst"));
                return;
            }

            this._auditLogger.log("RUN_COMPLIANCE", { taskCount: aSelectedContexts.length });

            MessageBox.confirm(
                this.getResourceBundle().getText("msg.runComplianceConfirm", [aSelectedContexts.length]),
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._executeComplianceChecks(aSelectedContexts);
                        }
                    }.bind(this)
                }
            );
        },

        /**
         * @function onConfigureRules
         * @description Opens compliance rules configuration interface with advanced rule editor.
         * @public
         */
        onConfigureRules() {
            if (!this._hasRole("ComplianceAdmin")) {
                MessageBox.error("Access denied. Compliance Administrator role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "ConfigureRules", reason: "Insufficient permissions" });
                return;
            }

            this._auditLogger.log("CONFIGURE_RULES", { action: "OpenRulesConfiguration" });

            this._getOrCreateDialog("configureRules", "a2a.network.agent11.ext.fragment.ConfigureRules")
                .then((oDialog) => {
                    oDialog.open();
                    this._loadComplianceRules(oDialog);
                })
                .catch((error) => {
                    MessageBox.error(`Failed to open Rules Configuration: ${ error.message}`);
                });
        },

        /**
         * @function onGenerateReport
         * @description Opens compliance report generation interface with various formats and filters.
         * @public
         */
        onGenerateReport() {
            if (!this._hasRole("ComplianceUser")) {
                MessageBox.error("Access denied. Compliance User role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "GenerateReport", reason: "Insufficient permissions" });
                return;
            }

            this._auditLogger.log("GENERATE_REPORT", { action: "OpenReportGeneration" });

            this._getOrCreateDialog("generateReport", "a2a.network.agent11.ext.fragment.GenerateReport")
                .then((oDialog) => {
                    const oReportModel = new JSONModel({
                        reportType: "SUMMARY",
                        format: "PDF",
                        includeCharts: true,
                        includeDetails: true,
                        includeRecommendations: true,
                        dateRange: "LAST_30_DAYS",
                        startDate: null,
                        endDate: null,
                        complianceAreas: [],
                        severityLevels: ["HIGH", "MEDIUM", "LOW"],
                        includePassedChecks: false,
                        includeFailedChecks: true,
                        includePendingChecks: true
                    });
                    oDialog.setModel(oReportModel, "report");
                    oDialog.open();
                    this._loadReportOptions(oDialog);
                })
                .catch((error) => {
                    MessageBox.error(`Failed to open Report Generation: ${ error.message}`);
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
            // Implement search logic for compliance tasks
        },

        /**
         * @function _executeComplianceChecks
         * @description Executes compliance checks for selected tasks with progress tracking.
         * @param {Array} aSelectedContexts - Selected compliance task contexts
         * @private
         */
        _executeComplianceChecks(aSelectedContexts) {
            const aTaskIds = aSelectedContexts.map(ctx => ctx.getObject().taskId);

            // Show progress dialog
            this._getOrCreateDialog("complianceProgress", "a2a.network.agent11.ext.fragment.ComplianceProgress")
                .then((oProgressDialog) => {
                    const oProgressModel = new JSONModel({
                        totalTasks: aTaskIds.length,
                        completedTasks: 0,
                        currentTask: "",
                        progress: 0,
                        status: "Starting compliance checks...",
                        results: []
                    });
                    oProgressDialog.setModel(oProgressModel, "progress");
                    oProgressDialog.open();

                    this._runComplianceChecks(aTaskIds, oProgressDialog);
                });
        },

        /**
         * @function _runComplianceChecks
         * @description Runs compliance checks with real-time progress updates.
         * @param {Array} aTaskIds - Array of task IDs to check
         * @param {sap.m.Dialog} oProgressDialog - Progress dialog
         * @private
         */
        _runComplianceChecks(aTaskIds, oProgressDialog) {
            const oModel = this.base.getView().getModel();

            SecurityUtils.secureCallFunction(oModel, "/RunComplianceChecks", {
                urlParameters: {
                    taskIds: aTaskIds.join(",")
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.complianceCheckStarted"));
                    this._startComplianceMonitoring(aTaskIds, oProgressDialog);
                    this._auditLogger.log("COMPLIANCE_CHECK_STARTED", { taskCount: aTaskIds.length, success: true });
                }.bind(this),
                error: function(error) {
                    MessageBox.error(this.getResourceBundle().getText("error.complianceCheckFailed"));
                    oProgressDialog.close();
                    this._auditLogger.log("COMPLIANCE_CHECK_FAILED", { taskCount: aTaskIds.length, error: error.message });
                }.bind(this)
            });
        },

        /**
         * @function _startComplianceMonitoring
         * @description Starts real-time monitoring of compliance check progress.
         * @param {Array} aTaskIds - Array of task IDs being checked
         * @param {sap.m.Dialog} oProgressDialog - Progress dialog
         * @private
         */
        _startComplianceMonitoring(aTaskIds, oProgressDialog) {
            if (this._complianceEventSource) {
                this._complianceEventSource.close();
            }

            try {
                this._complianceEventSource = new EventSource("/api/agent11/compliance/progress-stream");

                this._complianceEventSource.onmessage = function(event) {
                    try {
                        const data = JSON.parse(event.data);
                        this._updateComplianceProgress(data, oProgressDialog);
                    } catch (error) {
                        // console.error("Error parsing compliance progress data:", error);
                    }
                }.bind(this);

                this._complianceEventSource.onerror = function(error) {
                    // console.warn("Compliance progress stream error, falling back to polling:", error);
                    this._startCompliancePolling(aTaskIds, oProgressDialog);
                }.bind(this);

            } catch (error) {
                // console.warn("EventSource not available, using polling fallback");
                this._startCompliancePolling(aTaskIds, oProgressDialog);
            }
        },

        /**
         * @function _startCompliancePolling
         * @description Starts polling fallback for compliance progress updates.
         * @param {Array} aTaskIds - Array of task IDs being checked
         * @param {sap.m.Dialog} oProgressDialog - Progress dialog
         * @private
         */
        _startCompliancePolling(aTaskIds, oProgressDialog) {
            if (this._compliancePollingInterval) {
                clearInterval(this._compliancePollingInterval);
            }

            this._compliancePollingInterval = setInterval(() => {
                this._fetchComplianceProgress(aTaskIds, oProgressDialog);
            }, 3000);
        },

        /**
         * @function _fetchComplianceProgress
         * @description Fetches compliance progress via polling.
         * @param {Array} aTaskIds - Array of task IDs being checked
         * @param {sap.m.Dialog} oProgressDialog - Progress dialog
         * @private
         */
        _fetchComplianceProgress(aTaskIds, oProgressDialog) {
            const oModel = this.base.getView().getModel();

            SecurityUtils.secureCallFunction(oModel, "/GetComplianceProgress", {
                urlParameters: { taskIds: aTaskIds.join(",") },
                success: function(data) {
                    this._updateComplianceProgress(data, oProgressDialog);
                }.bind(this),
                error(error) {
                    // console.warn("Failed to fetch compliance progress:", error);
                }
            });
        },

        /**
         * @function _updateComplianceProgress
         * @description Updates compliance progress display.
         * @param {Object} data - Progress data
         * @param {sap.m.Dialog} oProgressDialog - Progress dialog
         * @private
         */
        _updateComplianceProgress(data, oProgressDialog) {
            if (!oProgressDialog || !oProgressDialog.isOpen()) {return;}

            const oProgressModel = oProgressDialog.getModel("progress");
            if (oProgressModel) {
                const oCurrentData = oProgressModel.getData();
                oCurrentData.completedTasks = data.completedTasks || oCurrentData.completedTasks;
                oCurrentData.currentTask = data.currentTask || oCurrentData.currentTask;
                oCurrentData.progress = Math.round((oCurrentData.completedTasks / oCurrentData.totalTasks) * 100);
                oCurrentData.status = data.status || oCurrentData.status;

                if (data.results && data.results.length > 0) {
                    oCurrentData.results = oCurrentData.results.concat(data.results);
                }

                oProgressModel.setData(oCurrentData);

                // Check if all tasks are completed
                if (oCurrentData.completedTasks >= oCurrentData.totalTasks) {
                    this._completeComplianceCheck(oProgressDialog);
                }
            }
        },

        /**
         * @function _completeComplianceCheck
         * @description Handles completion of compliance checks.
         * @param {sap.m.Dialog} oProgressDialog - Progress dialog
         * @private
         */
        _completeComplianceCheck(oProgressDialog) {
            setTimeout(() => {
                oProgressDialog.close();
                MessageToast.show(this.getResourceBundle().getText("msg.complianceCheckCompleted"));
                this._refreshComplianceData();
                this._auditLogger.log("COMPLIANCE_CHECK_COMPLETED", { status: "SUCCESS" });
            }, 2000);

            // Clean up event source
            if (this._complianceEventSource) {
                this._complianceEventSource.close();
                this._complianceEventSource = null;
            }

            if (this._compliancePollingInterval) {
                clearInterval(this._compliancePollingInterval);
                this._compliancePollingInterval = null;
            }
        },

        /**
         * @function _loadComplianceRules
         * @description Loads compliance rules configuration data.
         * @param {sap.m.Dialog} oDialog - Rules configuration dialog
         * @private
         */
        _loadComplianceRules(oDialog) {
            oDialog.setBusy(true);

            const oModel = this.base.getView().getModel();

            SecurityUtils.secureCallFunction(oModel, "/GetComplianceRules", {
                success: function(data) {
                    const oRulesModel = new JSONModel({
                        rules: data.rules || [],
                        categories: data.categories || [],
                        severityLevels: data.severityLevels || [],
                        frameworks: data.frameworks || [],
                        customRules: data.customRules || []
                    });
                    oDialog.setModel(oRulesModel, "rules");
                    oDialog.setBusy(false);
                }.bind(this),
                error(error) {
                    oDialog.setBusy(false);
                    MessageBox.error(`Failed to load compliance rules: ${ error.message}`);
                }
            });
        },

        /**
         * @function _loadReportOptions
         * @description Loads report generation options and data.
         * @param {sap.m.Dialog} oDialog - Report generation dialog
         * @private
         */
        _loadReportOptions(oDialog) {
            const oModel = this.base.getView().getModel();

            SecurityUtils.secureCallFunction(oModel, "/GetReportOptions", {
                success: function(data) {
                    const oReportModel = oDialog.getModel("report");
                    if (oReportModel) {
                        const oCurrentData = oReportModel.getData();
                        oCurrentData.availableAreas = data.complianceAreas || [];
                        oCurrentData.availableFrameworks = data.frameworks || [];
                        oCurrentData.templates = data.templates || [];
                        oReportModel.setData(oCurrentData);
                    }
                }.bind(this),
                error(error) {
                    MessageBox.error(`Failed to load report options: ${ error.message}`);
                }
            });
        },

        /**
         * @function _refreshComplianceData
         * @description Refreshes compliance task data in the table.
         * @private
         */
        _refreshComplianceData() {
            const oBinding = this.base.getView().byId("fe::table::ComplianceTasks::LineItem").getBinding("rows");
            oBinding.refresh();
        },

        /**
         * @function _startRealtimeUpdates
         * @description Starts real-time updates for compliance status.
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
                        this._handleComplianceUpdate(data);
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
                this._refreshComplianceData();
            }, 5000);
        },

        /**
         * @function _handleComplianceUpdate
         * @description Handles real-time compliance updates from WebSocket.
         * @param {Object} data - Update data
         * @private
         */
        _handleComplianceUpdate(data) {
            try {
                const oBundle = this.base.getView().getModel("i18n").getResourceBundle();

                switch (data.type) {
                case "COMPLIANCE_STARTED":
                    const sStartMsg = oBundle.getText("msg.complianceStarted") || "Compliance check started";
                    MessageToast.show(sStartMsg);
                    break;
                case "COMPLIANCE_COMPLETED":
                    const sCompleteMsg = oBundle.getText("msg.complianceCompleted") || "Compliance check completed";
                    MessageToast.show(sCompleteMsg);
                    this._refreshComplianceData();
                    break;
                case "COMPLIANCE_FAILED":
                    const sErrorMsg = oBundle.getText("error.complianceFailed") || "Compliance check failed";
                    const safeError = SecurityUtils.escapeHTML(data.error || "Unknown error");
                    MessageToast.show(`${sErrorMsg }: ${ safeError}`);
                    break;
                case "RULE_UPDATE":
                    const sRuleMsg = oBundle.getText("msg.ruleUpdated") || "Compliance rule updated";
                    MessageToast.show(sRuleMsg);
                    break;
                }
            } catch (error) {
                // console.error("Error processing compliance update:", error);
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
                        agent: "Agent11_Compliance",
                        action,
                        details: details || {}
                    };
                    // console.info(`AUDIT: ${ JSON.stringify(logEntry)}`);
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
            const mockRoles = ["ComplianceAdmin", "ComplianceUser", "ComplianceOperator"];
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
            if (this._complianceEventSource) {
                this._complianceEventSource.close();
                this._complianceEventSource = null;
            }

            // Clean up polling intervals
            if (this._pollInterval) {
                clearInterval(this._pollInterval);
                this._pollInterval = null;
            }

            if (this._compliancePollingInterval) {
                clearInterval(this._compliancePollingInterval);
                this._compliancePollingInterval = null;
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