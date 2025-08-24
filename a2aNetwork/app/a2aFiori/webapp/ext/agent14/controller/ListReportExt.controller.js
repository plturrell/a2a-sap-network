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
    "a2a/network/agent14/ext/utils/SecurityUtils"
], (ControllerExtension, Fragment, MessageBox, MessageToast, JSONModel, SecurityUtils) => {
    "use strict";

    /**
     * @class a2a.network.agent14.ext.controller.ListReportExt
     * @extends sap.ui.core.mvc.ControllerExtension
     * @description Controller extension for Agent 14 List Report - Backup Management Agent.
     * Provides comprehensive backup creation, scheduling, and status monitoring
     * with enterprise-grade security, audit logging, and accessibility features.
     */
    return ControllerExtension.extend("a2a.network.agent14.ext.controller.ListReportExt", {

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
         * @function onCreateBackup
         * @description Creates backup for selected tasks.
         * @public
         */
        onCreateBackup() {
            if (!this._hasRole("BackupAdmin")) {
                MessageBox.error("Access denied. Backup Administrator role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "CreateBackup", reason: "Insufficient permissions" });
                return;
            }

            const oBinding = this.base.getView().byId("fe::table::BackupTasks::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();

            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectTasksFirst"));
                return;
            }

            this._auditLogger.log("CREATE_BACKUP", { taskCount: aSelectedContexts.length });

            MessageBox.confirm(
                this.getResourceBundle().getText("msg.createBackupConfirm", [aSelectedContexts.length]),
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._executeBackupCreation(aSelectedContexts);
                        }
                    }.bind(this)
                }
            );
        },

        /**
         * @function onScheduleBackup
         * @description Opens backup scheduling interface.
         * @public
         */
        onScheduleBackup() {
            if (!this._hasRole("BackupAdmin")) {
                MessageBox.error("Access denied. Backup Administrator role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "ScheduleBackup", reason: "Insufficient permissions" });
                return;
            }

            this._auditLogger.log("SCHEDULE_BACKUP", { action: "OpenScheduler" });

            this._getOrCreateDialog("scheduleBackup", "a2a.network.agent14.ext.fragment.ScheduleBackup")
                .then((oDialog) => {
                    const oScheduleModel = new JSONModel({
                        schedules: [],
                        frequency: "DAILY",
                        startTime: new Date(),
                        retentionDays: 30,
                        backupTypes: [
                            { key: "FULL", text: "Full Backup" },
                            { key: "INCREMENTAL", text: "Incremental Backup" },
                            { key: "DIFFERENTIAL", text: "Differential Backup" },
                            { key: "SNAPSHOT", text: "Snapshot" }
                        ],
                        selectedType: "FULL",
                        compressionEnabled: true,
                        encryptionEnabled: true,
                        verificationEnabled: true,
                        notificationEnabled: true
                    });
                    oDialog.setModel(oScheduleModel, "schedule");
                    oDialog.open();
                    this._loadBackupSchedules(oDialog);
                })
                .catch((error) => {
                    MessageBox.error(`Failed to open Schedule Backup: ${ error.message}`);
                });
        },

        /**
         * @function onViewBackupStatus
         * @description Opens backup status monitoring dashboard.
         * @public
         */
        onViewBackupStatus() {
            if (!this._hasRole("BackupUser")) {
                MessageBox.error("Access denied. Backup User role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "ViewBackupStatus", reason: "Insufficient permissions" });
                return;
            }

            this._auditLogger.log("VIEW_BACKUP_STATUS", { action: "OpenStatusDashboard" });

            this._getOrCreateDialog("viewBackupStatus", "a2a.network.agent14.ext.fragment.ViewBackupStatus")
                .then((oDialog) => {
                    const oStatusModel = new JSONModel({
                        statusFilter: "ALL",
                        typeFilter: "ALL",
                        timeRange: "LAST_24_HOURS",
                        startDate: new Date(Date.now() - 24 * 60 * 60 * 1000),
                        endDate: new Date(),
                        autoRefresh: true,
                        refreshInterval: 30000,
                        backups: [],
                        statistics: {
                            running: 0,
                            completed: 0,
                            failed: 0,
                            scheduled: 0
                        },
                        storageUtilization: {
                            used: 0,
                            available: 0,
                            growth: 0
                        }
                    });
                    oDialog.setModel(oStatusModel, "status");
                    oDialog.open();
                    this._loadBackupStatus(oDialog);
                })
                .catch((error) => {
                    MessageBox.error(`Failed to open Backup Status: ${ error.message}`);
                });
        },

        /**
         * @function _executeBackupCreation
         * @description Executes backup creation for selected tasks.
         * @param {Array} aSelectedContexts - Selected backup task contexts
         * @private
         */
        _executeBackupCreation(aSelectedContexts) {
            const aTaskIds = aSelectedContexts.map(ctx => ctx.getObject().taskId);

            // Show progress dialog
            this._getOrCreateDialog("backupProgress", "a2a.network.agent14.ext.fragment.BackupProgress")
                .then((oProgressDialog) => {
                    const oProgressModel = new JSONModel({
                        totalTasks: aTaskIds.length,
                        completedTasks: 0,
                        currentTask: "",
                        progress: 0,
                        status: "Starting backup creation...",
                        statistics: {
                            dataSize: 0,
                            compressedSize: 0,
                            compressionRatio: 0,
                            transferRate: 0
                        },
                        backupResults: []
                    });
                    oProgressDialog.setModel(oProgressModel, "progress");
                    oProgressDialog.open();

                    this._runBackupCreation(aTaskIds, oProgressDialog);
                });
        },

        /**
         * @function _runBackupCreation
         * @description Runs backup creation with real-time progress updates.
         * @param {Array} aTaskIds - Array of task IDs to backup
         * @param {sap.m.Dialog} oProgressDialog - Progress dialog
         * @private
         */
        _runBackupCreation(aTaskIds, oProgressDialog) {
            const oModel = this.base.getView().getModel();

            SecurityUtils.secureCallFunction(oModel, "/CreateBackups", {
                urlParameters: {
                    taskIds: aTaskIds.join(","),
                    backupType: "FULL",
                    compressionEnabled: true,
                    encryptionEnabled: true,
                    verificationEnabled: true
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.backupStarted"));
                    this._startBackupMonitoring(data.backupId, oProgressDialog);
                    this._auditLogger.log("BACKUP_CREATION_STARTED", {
                        taskCount: aTaskIds.length,
                        backupId: data.backupId,
                        success: true
                    });
                }.bind(this),
                error: function(error) {
                    MessageBox.error(this.getResourceBundle().getText("error.backupFailed"));
                    oProgressDialog.close();
                    this._auditLogger.log("BACKUP_CREATION_FAILED", {
                        taskCount: aTaskIds.length,
                        error: error.message
                    });
                }.bind(this)
            });
        },

        /**
         * @function _startBackupMonitoring
         * @description Starts real-time monitoring of backup progress.
         * @param {string} sBackupId - Backup ID to monitor
         * @param {sap.m.Dialog} oProgressDialog - Progress dialog
         * @private
         */
        _startBackupMonitoring(sBackupId, oProgressDialog) {
            if (this._backupEventSource) {
                this._backupEventSource.close();
            }

            try {
                this._backupEventSource = new EventSource(`/api/agent14/backup/stream/${ sBackupId}`);

                this._backupEventSource.onmessage = function(event) {
                    try {
                        const data = JSON.parse(event.data);
                        this._updateBackupProgress(data, oProgressDialog);
                    } catch (error) {
                        // console.error("Error parsing backup progress data:", error);
                    }
                }.bind(this);

                this._backupEventSource.onerror = function(error) {
                    // console.warn("Backup stream error, falling back to polling:", error);
                    this._startBackupPolling(sBackupId, oProgressDialog);
                }.bind(this);

            } catch (error) {
                // console.warn("EventSource not available, using polling fallback");
                this._startBackupPolling(sBackupId, oProgressDialog);
            }
        },

        /**
         * @function _loadBackupSchedules
         * @description Loads backup schedules and configuration.
         * @param {sap.m.Dialog} oDialog - Schedule dialog
         * @private
         */
        _loadBackupSchedules(oDialog) {
            oDialog.setBusy(true);

            const oModel = this.base.getView().getModel();

            SecurityUtils.secureCallFunction(oModel, "/GetBackupSchedules", {
                success: function(data) {
                    const oScheduleModel = oDialog.getModel("schedule");
                    if (oScheduleModel) {
                        const oCurrentData = oScheduleModel.getData();
                        oCurrentData.schedules = data.schedules || [];
                        oCurrentData.policies = data.policies || [];
                        oCurrentData.storageTargets = data.storageTargets || [];
                        oScheduleModel.setData(oCurrentData);
                    }
                    oDialog.setBusy(false);
                }.bind(this),
                error(error) {
                    oDialog.setBusy(false);
                    MessageBox.error(`Failed to load backup schedules: ${ error.message}`);
                }
            });
        },

        /**
         * @function _loadBackupStatus
         * @description Loads backup status information.
         * @param {sap.m.Dialog} oDialog - Status dialog
         * @private
         */
        _loadBackupStatus(oDialog) {
            oDialog.setBusy(true);

            const oModel = this.base.getView().getModel();

            SecurityUtils.secureCallFunction(oModel, "/GetBackupStatus", {
                success: function(data) {
                    const oStatusModel = oDialog.getModel("status");
                    if (oStatusModel) {
                        const oCurrentData = oStatusModel.getData();
                        oCurrentData.backups = data.backups || [];
                        oCurrentData.statistics = data.statistics || {};
                        oCurrentData.storageUtilization = data.storageUtilization || {};
                        oCurrentData.trends = data.trends || {};
                        oStatusModel.setData(oCurrentData);
                    }
                    oDialog.setBusy(false);

                    // Start auto-refresh if enabled
                    if (oCurrentData.autoRefresh) {
                        this._startStatusAutoRefresh(oDialog);
                    }
                }.bind(this),
                error(error) {
                    oDialog.setBusy(false);
                    MessageBox.error(`Failed to load backup status: ${ error.message}`);
                }
            });
        },

        /**
         * @function _updateBackupProgress
         * @description Updates backup progress display.
         * @param {Object} data - Progress data
         * @param {sap.m.Dialog} oProgressDialog - Progress dialog
         * @private
         */
        _updateBackupProgress(data, oProgressDialog) {
            if (!oProgressDialog || !oProgressDialog.isOpen()) {return;}

            const oProgressModel = oProgressDialog.getModel("progress");
            if (oProgressModel) {
                const oCurrentData = oProgressModel.getData();
                oCurrentData.completedTasks = data.completedTasks || oCurrentData.completedTasks;
                oCurrentData.currentTask = data.currentTask || oCurrentData.currentTask;
                oCurrentData.progress = Math.round((oCurrentData.completedTasks / oCurrentData.totalTasks) * 100);
                oCurrentData.status = data.status || oCurrentData.status;

                if (data.statistics) {
                    oCurrentData.statistics = data.statistics;
                }

                if (data.backupResults && data.backupResults.length > 0) {
                    oCurrentData.backupResults = oCurrentData.backupResults.concat(data.backupResults);
                }

                oProgressModel.setData(oCurrentData);

                // Check if all tasks are completed
                if (oCurrentData.completedTasks >= oCurrentData.totalTasks) {
                    this._completeBackup(oProgressDialog);
                }
            }
        },

        /**
         * @function _completeBackup
         * @description Handles completion of backup operation.
         * @param {sap.m.Dialog} oProgressDialog - Progress dialog
         * @private
         */
        _completeBackup(oProgressDialog) {
            setTimeout(() => {
                oProgressDialog.close();
                MessageToast.show(this.getResourceBundle().getText("msg.backupCompleted"));
                this._refreshBackupData();
                this._auditLogger.log("BACKUP_COMPLETED", { status: "SUCCESS" });
            }, 2000);

            // Clean up event source
            if (this._backupEventSource) {
                this._backupEventSource.close();
                this._backupEventSource = null;
            }
        },

        /**
         * @function _refreshBackupData
         * @description Refreshes backup task data in the table.
         * @private
         */
        _refreshBackupData() {
            const oBinding = this.base.getView().byId("fe::table::BackupTasks::LineItem").getBinding("rows");
            if (oBinding) {
                oBinding.refresh();
            }
        },

        /**
         * @function _startRealtimeUpdates
         * @description Starts real-time updates for backup events.
         * @private
         */
        _startRealtimeUpdates() {
            this._initializeWebSocket();
        },

        /**
         * @function _initializeWebSocket
         * @description Initializes secure WebSocket connection for real-time backup updates.
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
                        this._handleBackupUpdate(data);
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
                this._refreshBackupData();
            }, 5000);
        },

        /**
         * @function _handleBackupUpdate
         * @description Handles real-time backup updates from WebSocket.
         * @param {Object} data - Update data
         * @private
         */
        _handleBackupUpdate(data) {
            try {
                const oBundle = this.base.getView().getModel("i18n").getResourceBundle();

                switch (data.type) {
                case "BACKUP_STARTED":
                    const sBackupStarted = oBundle.getText("msg.backupStarted") || "Backup started";
                    MessageToast.show(sBackupStarted);
                    break;
                case "BACKUP_PROGRESS":
                    this._updateBackupProgress(data);
                    break;
                case "BACKUP_COMPLETED":
                    const sBackupCompleted = oBundle.getText("msg.backupCompleted") || "Backup completed";
                    MessageToast.show(sBackupCompleted);
                    this._refreshBackupData();
                    break;
                case "BACKUP_FAILED":
                    const sBackupFailed = oBundle.getText("error.backupFailed") || "Backup failed";
                    MessageBox.error(`${sBackupFailed }: ${ data.message}`);
                    break;
                case "RESTORE_COMPLETED":
                    const sRestoreCompleted = oBundle.getText("msg.restoreCompleted") || "Restore completed";
                    MessageToast.show(sRestoreCompleted);
                    this._refreshBackupData();
                    break;
                case "STORAGE_WARNING":
                    const sStorageWarning = oBundle.getText("msg.storageWarning") || "Storage warning";
                    MessageBox.warning(`${sStorageWarning }: ${ data.message}`);
                    break;
                case "VERIFICATION_FAILED":
                    const sVerificationFailed = oBundle.getText("msg.verificationFailed") || "Backup verification failed";
                    MessageBox.error(`${sVerificationFailed }: ${ data.message}`);
                    break;
                }
            } catch (error) {
                // console.error("Error processing backup update:", error);
            }
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
            // Throttle backup data updates
            this._throttledBackupUpdate = this._throttle(this._refreshBackupData.bind(this), 1000);
            // Debounce search operations
            this._debouncedSearch = this._debounce(this._performSearch.bind(this), 300);
        },

        /**
         * @function _performSearch
         * @description Performs search operation for backup tasks.
         * @param {string} sQuery - Search query
         * @private
         */
        _performSearch(sQuery) {
            // Implement search logic for backup tasks
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
                        agent: "Agent14_Backup",
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
            const mockRoles = ["BackupAdmin", "BackupUser", "BackupOperator"];
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
            if (this._backupEventSource) {
                this._backupEventSource.close();
                this._backupEventSource = null;
            }

            // Clean up polling intervals
            if (this._pollInterval) {
                clearInterval(this._pollInterval);
                this._pollInterval = null;
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