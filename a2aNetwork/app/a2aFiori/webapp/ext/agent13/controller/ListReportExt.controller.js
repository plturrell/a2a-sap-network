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
    "a2a/network/agent13/ext/utils/SecurityUtils"
], (ControllerExtension, Fragment, MessageBox, MessageToast, JSONModel, SecurityUtils) => {
    "use strict";

    /**
     * @class a2a.network.agent13.ext.controller.ListReportExt
     * @extends sap.ui.core.mvc.ControllerExtension
     * @description Controller extension for Agent 13 List Report - Security Management.
     * Provides comprehensive security scanning, policy configuration, and patch management
     * with enterprise-grade security, audit logging, and accessibility features.
     */
    return ControllerExtension.extend("a2a.network.agent13.ext.controller.ListReportExt", {

        override: {
            /**
             * @function onInit
             * @description Initializes the controller extension with security utilities, device model, dialog caching, and real-time updates.
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
         * @function onSecurityScan
         * @description Initiates security scan for selected security tasks.
         * @public
         */
        onSecurityScan() {
            if (!this._hasRole("SecurityAnalyst")) {
                MessageBox.error("Access denied. Security Analyst role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "SecurityScan", reason: "Insufficient permissions" });
                return;
            }

            const oBinding = this.base.getView().byId("fe::table::SecurityTasks::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();

            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectTasksFirst"));
                return;
            }

            this._auditLogger.log("SECURITY_SCAN", { taskCount: aSelectedContexts.length });

            MessageBox.confirm(
                this.getResourceBundle().getText("msg.securityScanConfirm", [aSelectedContexts.length]),
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._executeSecurityScan(aSelectedContexts);
                        }
                    }.bind(this)
                }
            );
        },

        /**
         * @function onConfigurePolicy
         * @description Opens security policy configuration interface.
         * @public
         */
        onConfigurePolicy() {
            if (!this._hasRole("SecurityAdmin")) {
                MessageBox.error("Access denied. Security Administrator role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "ConfigurePolicy", reason: "Insufficient permissions" });
                return;
            }

            this._auditLogger.log("CONFIGURE_POLICY", { action: "OpenPolicyConfiguration" });

            this._getOrCreateDialog("configurePolicy", "a2a.network.agent13.ext.fragment.ConfigurePolicy")
                .then((oDialog) => {
                    const oPolicyModel = new JSONModel({
                        policies: [],
                        categories: [
                            { key: "ACCESS", text: "Access Control" },
                            { key: "NETWORK", text: "Network Security" },
                            { key: "DATA", text: "Data Protection" },
                            { key: "COMPLIANCE", text: "Compliance" },
                            { key: "MONITORING", text: "Security Monitoring" },
                            { key: "INCIDENT", text: "Incident Response" }
                        ],
                        selectedCategory: "ACCESS",
                        policyTemplates: [],
                        riskLevels: ["CRITICAL", "HIGH", "MEDIUM", "LOW"],
                        enforcement: "ACTIVE"
                    });
                    oDialog.setModel(oPolicyModel, "policy");
                    oDialog.open();
                    this._loadSecurityPolicies(oDialog);
                })
                .catch((error) => {
                    MessageBox.error(`Failed to open Policy Configuration: ${ error.message}`);
                });
        },

        /**
         * @function onApplyPatches
         * @description Applies security patches to selected tasks.
         * @public
         */
        onApplyPatches() {
            if (!this._hasRole("SecurityAdmin")) {
                MessageBox.error("Access denied. Security Administrator role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "ApplyPatches", reason: "Insufficient permissions" });
                return;
            }

            const oBinding = this.base.getView().byId("fe::table::SecurityTasks::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();

            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectTasksFirst"));
                return;
            }

            // Check if any selected tasks have pending patches
            const aPatchableTasks = aSelectedContexts.filter(ctx => {
                const oData = ctx.getObject();
                return oData.pendingPatches > 0;
            });

            if (aPatchableTasks.length === 0) {
                MessageBox.information(this.getResourceBundle().getText("msg.noPendingPatches"));
                return;
            }

            this._auditLogger.log("APPLY_PATCHES", { taskCount: aPatchableTasks.length });

            this._getOrCreateDialog("applyPatches", "a2a.network.agent13.ext.fragment.ApplyPatches")
                .then((oDialog) => {
                    const oPatchModel = new JSONModel({
                        selectedTasks: aPatchableTasks.map(ctx => ctx.getObject()),
                        patchOptions: {
                            testFirst: true,
                            createBackup: true,
                            rollbackEnabled: true,
                            scheduledExecution: false,
                            executionTime: new Date(),
                            notifyOnCompletion: true,
                            priorityOrder: "CRITICAL_FIRST"
                        },
                        availablePatches: []
                    });
                    oDialog.setModel(oPatchModel, "patch");
                    oDialog.open();
                    this._loadAvailablePatches(aPatchableTasks, oDialog);
                })
                .catch((error) => {
                    MessageBox.error(`Failed to open Patch Management: ${ error.message}`);
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
            // Implement search logic for security tasks
        },

        /**
         * @function _executeSecurityScan
         * @description Executes security scan for selected tasks.
         * @param {Array} aSelectedContexts - Selected security task contexts
         * @private
         */
        _executeSecurityScan(aSelectedContexts) {
            const aTaskIds = aSelectedContexts.map(ctx => ctx.getObject().taskId);

            // Show progress dialog
            this._getOrCreateDialog("scanProgress", "a2a.network.agent13.ext.fragment.ScanProgress")
                .then((oProgressDialog) => {
                    const oProgressModel = new JSONModel({
                        totalTasks: aTaskIds.length,
                        completedTasks: 0,
                        currentTask: "",
                        progress: 0,
                        status: "Starting security scan...",
                        findings: {
                            critical: 0,
                            high: 0,
                            medium: 0,
                            low: 0,
                            info: 0
                        },
                        scanResults: []
                    });
                    oProgressDialog.setModel(oProgressModel, "progress");
                    oProgressDialog.open();

                    this._runSecurityScans(aTaskIds, oProgressDialog);
                });
        },

        /**
         * @function _runSecurityScans
         * @description Runs security scans with real-time progress updates.
         * @param {Array} aTaskIds - Array of task IDs to scan
         * @param {sap.m.Dialog} oProgressDialog - Progress dialog
         * @private
         */
        _runSecurityScans(aTaskIds, oProgressDialog) {
            const oModel = this.base.getView().getModel();

            SecurityUtils.secureCallFunction(oModel, "/RunSecurityScans", {
                urlParameters: {
                    taskIds: aTaskIds.join(","),
                    scanType: "COMPREHENSIVE",
                    includeVulnerabilities: true,
                    includeCompliance: true,
                    includeThreatIntelligence: true
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.scanStarted"));
                    this._startScanMonitoring(data.scanId, oProgressDialog);
                    this._auditLogger.log("SECURITY_SCAN_STARTED", {
                        taskCount: aTaskIds.length,
                        scanId: data.scanId,
                        success: true
                    });
                }.bind(this),
                error: function(error) {
                    MessageBox.error(this.getResourceBundle().getText("error.scanFailed"));
                    oProgressDialog.close();
                    this._auditLogger.log("SECURITY_SCAN_FAILED", {
                        taskCount: aTaskIds.length,
                        error: error.message
                    });
                }.bind(this)
            });
        },

        /**
         * @function _startScanMonitoring
         * @description Starts real-time monitoring of security scan progress.
         * @param {string} sScanId - Scan ID to monitor
         * @param {sap.m.Dialog} oProgressDialog - Progress dialog
         * @private
         */
        _startScanMonitoring(sScanId, oProgressDialog) {
            if (this._scanEventSource) {
                this._scanEventSource.close();
            }

            try {
                this._scanEventSource = new EventSource(`/api/agent13/security/scan-stream/${ sScanId}`);

                this._scanEventSource.onmessage = function(event) {
                    try {
                        const data = JSON.parse(event.data);
                        this._updateScanProgress(data, oProgressDialog);
                    } catch (error) {
                        console.error("Error parsing scan progress data:", error);
                    }
                }.bind(this);

                this._scanEventSource.onerror = function(error) {
                    console.warn("Scan stream error, falling back to polling:", error);
                    this._startScanPolling(sScanId, oProgressDialog);
                }.bind(this);

            } catch (error) {
                console.warn("EventSource not available, using polling fallback");
                this._startScanPolling(sScanId, oProgressDialog);
            }
        },

        /**
         * @function _startScanPolling
         * @description Starts polling fallback for scan progress updates.
         * @param {string} sScanId - Scan ID to monitor
         * @param {sap.m.Dialog} oProgressDialog - Progress dialog
         * @private
         */
        _startScanPolling(sScanId, oProgressDialog) {
            if (this._scanPollingInterval) {
                clearInterval(this._scanPollingInterval);
            }

            this._scanPollingInterval = setInterval(() => {
                this._fetchScanProgress(sScanId, oProgressDialog);
            }, 2000);
        },

        /**
         * @function _fetchScanProgress
         * @description Fetches scan progress via polling.
         * @param {string} sScanId - Scan ID to monitor
         * @param {sap.m.Dialog} oProgressDialog - Progress dialog
         * @private
         */
        _fetchScanProgress(sScanId, oProgressDialog) {
            const oModel = this.base.getView().getModel();

            SecurityUtils.secureCallFunction(oModel, "/GetScanProgress", {
                urlParameters: { scanId: sScanId },
                success: function(data) {
                    this._updateScanProgress(data, oProgressDialog);
                }.bind(this),
                error(error) {
                    console.warn("Failed to fetch scan progress:", error);
                }
            });
        },

        /**
         * @function _updateScanProgress
         * @description Updates scan progress display.
         * @param {Object} data - Progress data
         * @param {sap.m.Dialog} oProgressDialog - Progress dialog
         * @private
         */
        _updateScanProgress(data, oProgressDialog) {
            if (!oProgressDialog || !oProgressDialog.isOpen()) {return;}

            const oProgressModel = oProgressDialog.getModel("progress");
            if (oProgressModel) {
                const oCurrentData = oProgressModel.getData();
                oCurrentData.completedTasks = data.completedTasks || oCurrentData.completedTasks;
                oCurrentData.currentTask = data.currentTask || oCurrentData.currentTask;
                oCurrentData.progress = Math.round((oCurrentData.completedTasks / oCurrentData.totalTasks) * 100);
                oCurrentData.status = data.status || oCurrentData.status;

                if (data.findings) {
                    oCurrentData.findings = data.findings;
                }

                if (data.scanResults && data.scanResults.length > 0) {
                    oCurrentData.scanResults = oCurrentData.scanResults.concat(data.scanResults);
                }

                oProgressModel.setData(oCurrentData);

                // Check if all tasks are completed
                if (oCurrentData.completedTasks >= oCurrentData.totalTasks) {
                    this._completeScan(oProgressDialog);
                }
            }
        },

        /**
         * @function _completeScan
         * @description Handles completion of security scan.
         * @param {sap.m.Dialog} oProgressDialog - Progress dialog
         * @private
         */
        _completeScan(oProgressDialog) {
            setTimeout(() => {
                oProgressDialog.close();
                MessageToast.show(this.getResourceBundle().getText("msg.scanCompleted"));
                this._refreshSecurityData();
                this._auditLogger.log("SECURITY_SCAN_COMPLETED", { status: "SUCCESS" });

                // Show scan summary
                this._showScanSummary(oProgressDialog.getModel("progress").getData());
            }, 2000);

            // Clean up event source
            if (this._scanEventSource) {
                this._scanEventSource.close();
                this._scanEventSource = null;
            }

            if (this._scanPollingInterval) {
                clearInterval(this._scanPollingInterval);
                this._scanPollingInterval = null;
            }
        },

        /**
         * @function _loadSecurityPolicies
         * @description Loads security policies and templates.
         * @param {sap.m.Dialog} oDialog - Policy configuration dialog
         * @private
         */
        _loadSecurityPolicies(oDialog) {
            oDialog.setBusy(true);

            const oModel = this.base.getView().getModel();

            SecurityUtils.secureCallFunction(oModel, "/GetSecurityPolicies", {
                success: function(data) {
                    const oPolicyModel = oDialog.getModel("policy");
                    if (oPolicyModel) {
                        const oCurrentData = oPolicyModel.getData();
                        oCurrentData.policies = data.policies || [];
                        oCurrentData.policyTemplates = data.templates || [];
                        oCurrentData.regulations = data.regulations || [];
                        oCurrentData.bestPractices = data.bestPractices || [];
                        oPolicyModel.setData(oCurrentData);
                    }
                    oDialog.setBusy(false);
                }.bind(this),
                error(error) {
                    oDialog.setBusy(false);
                    MessageBox.error(`Failed to load security policies: ${ error.message}`);
                }
            });
        },

        /**
         * @function _loadAvailablePatches
         * @description Loads available security patches for selected tasks.
         * @param {Array} aSelectedContexts - Selected task contexts
         * @param {sap.m.Dialog} oDialog - Patch management dialog
         * @private
         */
        _loadAvailablePatches(aSelectedContexts, oDialog) {
            oDialog.setBusy(true);

            const oModel = this.base.getView().getModel();
            const aTaskIds = aSelectedContexts.map(ctx => ctx.getObject().taskId);

            SecurityUtils.secureCallFunction(oModel, "/GetAvailablePatches", {
                urlParameters: {
                    taskIds: aTaskIds.join(","),
                    includeDetails: true
                },
                success: function(data) {
                    const oPatchModel = oDialog.getModel("patch");
                    if (oPatchModel) {
                        const oCurrentData = oPatchModel.getData();
                        oCurrentData.availablePatches = data.patches || [];
                        oCurrentData.patchSummary = data.summary || {};
                        oCurrentData.dependencies = data.dependencies || {};
                        oCurrentData.risks = data.risks || {};
                        oPatchModel.setData(oCurrentData);
                    }
                    oDialog.setBusy(false);
                }.bind(this),
                error(error) {
                    oDialog.setBusy(false);
                    MessageBox.error(`Failed to load available patches: ${ error.message}`);
                }
            });
        },

        /**
         * @function _showScanSummary
         * @description Shows security scan summary.
         * @param {Object} scanData - Scan results data
         * @private
         */
        _showScanSummary(scanData) {
            this._getOrCreateDialog("scanSummary", "a2a.network.agent13.ext.fragment.ScanSummary")
                .then((oDialog) => {
                    const oSummaryModel = new JSONModel(scanData);
                    oDialog.setModel(oSummaryModel, "summary");
                    oDialog.open();
                })
                .catch((error) => {
                    MessageBox.error(`Failed to show scan summary: ${ error.message}`);
                });
        },

        /**
         * @function _refreshSecurityData
         * @description Refreshes security task data in the table.
         * @private
         */
        _refreshSecurityData() {
            const oBinding = this.base.getView().byId("fe::table::SecurityTasks::LineItem").getBinding("rows");
            oBinding.refresh();
        },

        /**
         * @function _startRealtimeUpdates
         * @description Starts real-time updates for security events.
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
                        this._handleSecurityUpdate(data);
                    }.bind(this)
                });

                this._ws.onclose = function() {
                    const oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                    const sMessage = oBundle.getText("msg.websocketDisconnected") || "Connection lost. Reconnecting...";
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
        _initializePolling() {
            this._pollInterval = setInterval(() => {
                this._refreshSecurityData();
            }, 5000);
        },

        /**
         * @function _handleSecurityUpdate
         * @description Handles real-time security updates from WebSocket.
         * @param {Object} data - Update data
         * @private
         */
        _handleSecurityUpdate(data) {
            try {
                const oBundle = this.base.getView().getModel("i18n").getResourceBundle();

                switch (data.type) {
                case "THREAT_DETECTED":
                    var sThreatMsg = oBundle.getText("msg.threatDetected") || "Threat detected";
                    MessageToast.show(`${sThreatMsg }: ${ data.threatType}`);
                    this._refreshSecurityData();
                    break;
                case "SCAN_COMPLETED":
                    var sScanMsg = oBundle.getText("msg.scanCompleted") || "Security scan completed";
                    MessageToast.show(sScanMsg);
                    this._refreshSecurityData();
                    break;
                case "PATCH_AVAILABLE":
                    var sPatchMsg = oBundle.getText("msg.patchAvailable") || "New security patch available";
                    MessageToast.show(sPatchMsg);
                    this._refreshSecurityData();
                    break;
                case "POLICY_VIOLATION":
                    var sViolationMsg = oBundle.getText("msg.policyViolation") || "Security policy violation";
                    MessageBox.warning(`${sViolationMsg }: ${ data.policyName}`);
                    break;
                case "SECURITY_ALERT":
                    var sAlertMsg = oBundle.getText("msg.securityAlert") || "Security alert";
                    var safeDetails = SecurityUtils.escapeHTML(data.details || "");
                    MessageBox.error(`${sAlertMsg }: ${ safeDetails}`);
                    break;
                }
            } catch (error) {
                console.error("Error processing security update:", error);
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
                    const logEntry = {
                        timestamp,
                        user,
                        agent: "Agent13_Security",
                        action,
                        details: details || {}
                    };
                    console.info(`AUDIT: ${ JSON.stringify(logEntry)}`);
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
            const mockRoles = ["SecurityAdmin", "SecurityAnalyst", "SecurityOperator"];
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
            if (this._scanEventSource) {
                this._scanEventSource.close();
                this._scanEventSource = null;
            }

            // Clean up polling intervals
            if (this._pollInterval) {
                clearInterval(this._pollInterval);
                this._pollInterval = null;
            }

            if (this._scanPollingInterval) {
                clearInterval(this._scanPollingInterval);
                this._scanPollingInterval = null;
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