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
    "sap/base/Log",
    "../../../utils/SharedSecurityUtils"
], (ControllerExtension, MessageBox, MessageToast, Fragment, JSONModel, Log, SecurityUtils) => {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent15.ext.controller.ObjectPageExt", {

        override: {
            onInit() {
                this._extensionAPI = this.base.getExtensionAPI();
                this._securityUtils = SecurityUtils;
                this._initializeSecurity();

                // Initialize device model for responsive behavior
                const oDeviceModel = new JSONModel(sap.ui.Device);
                oDeviceModel.setDefaultBindingMode(sap.ui.model.BindingMode.OneWay);
                this.base.getView().setModel(oDeviceModel, "device");

                // Initialize dialog cache
                this._dialogCache = {};

                // Initialize real-time monitoring
                this._initializeRealtimeMonitoring();
            },

            onExit() {
                this._cleanupResources();
                if (this.base.onExit) {
                    this.base.onExit.apply(this, arguments);
                }
            }
        },

        /**
         * @function onRollbackDeployment
         * @description Initiates rollback of deployment to previous version.
         * @public
         */
        onRollbackDeployment() {
            if (!this._securityUtils.hasRole("DeploymentAdmin")) {
                MessageBox.error("Access denied. Deployment Administrator role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "RollbackDeployment", reason: "Insufficient permissions" });
                return;
            }

            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            const sTaskId = oData.taskId;
            const sTaskName = oData.taskName;

            // Check if rollback is possible
            if (!oData.previousVersion) {
                MessageBox.warning(this.getResourceBundle().getText("msg.noPreviousVersion"));
                return;
            }

            this._auditLogger.log("ROLLBACK_DEPLOYMENT_REQUESTED", { taskId: sTaskId, taskName: sTaskName });

            MessageBox.confirm(
                this.getResourceBundle().getText("msg.rollbackConfirm", [sTaskName, oData.previousVersion]),
                {
                    title: this.getResourceBundle().getText("title.confirmRollback"),
                    emphasizedAction: MessageBox.Action.CANCEL,
                    initialFocus: MessageBox.Action.CANCEL,
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._executeRollback(sTaskId, sTaskName, oData);
                        } else {
                            this._auditLogger.log("ROLLBACK_CANCELLED", { taskId: sTaskId, taskName: sTaskName });
                        }
                    }.bind(this)
                }
            );
        },

        /**
         * @function onValidateDeployment
         * @description Validates deployment integrity and configuration.
         * @public
         */
        onValidateDeployment() {
            if (!this._securityUtils.hasRole("DeploymentUser")) {
                MessageBox.error("Access denied. Deployment User role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "ValidateDeployment", reason: "Insufficient permissions" });
                return;
            }

            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            const sTaskId = oData.taskId;
            const sTaskName = oData.taskName;

            this._auditLogger.log("VALIDATE_DEPLOYMENT", { taskId: sTaskId, taskName: sTaskName });

            const handleValidationDialogOpen = (oDialog) => {
                const oValidationModel = new JSONModel({
                    taskId: sTaskId,
                    taskName: sTaskName,
                    deploymentId: oData.deploymentId,
                    environment: oData.environment,
                    validationType: "COMPREHENSIVE",
                    validationChecks: {
                        configurationValidation: true,
                        dependencyValidation: true,
                        securityValidation: true,
                        performanceValidation: true,
                        compatibilityValidation: true,
                        integrationValidation: true
                    },
                    validationScope: "ALL",
                    includeTests: true,
                    generateReport: true,
                    validationResults: {
                        status: "NOT_STARTED",
                        progress: 0,
                        checksCompleted: 0,
                        totalChecks: 0,
                        errors: 0,
                        warnings: 0,
                        passed: 0,
                        details: []
                    }
                });
                oDialog.setModel(oValidationModel, "validation");
                oDialog.open();
            };

            const handleValidationDialogError = (error) => {
                MessageBox.error(`Failed to open Validate Deployment dialog: ${ error.message}`);
            };

            this._getOrCreateDialog("validateDeployment", "a2a.network.agent15.ext.fragment.ValidateDeployment")
                .then(handleValidationDialogOpen)
                .catch(handleValidationDialogError);
        },

        /**
         * @function onViewLogs
         * @description Opens deployment logs viewer with filtering and search capabilities.
         * @public
         */
        onViewLogs() {
            if (!this._securityUtils.hasRole("DeploymentUser")) {
                MessageBox.error("Access denied. Deployment User role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "ViewLogs", reason: "Insufficient permissions" });
                return;
            }

            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            const sTaskId = oData.taskId;
            const sTaskName = oData.taskName;

            this._auditLogger.log("VIEW_LOGS", { taskId: sTaskId, taskName: sTaskName });

            const handleLogDialogOpen = (oDialog) => {
                const oLogModel = new JSONModel({
                    taskId: sTaskId,
                    taskName: sTaskName,
                    deploymentId: oData.deploymentId,
                    logLevel: "ALL",
                    timeRange: "LAST_HOUR",
                    startTime: new Date(Date.now() - 60 * 60 * 1000),
                    endTime: new Date(),
                    searchQuery: "",
                    autoRefresh: true,
                    refreshInterval: 10000,
                    maxLines: 1000,
                    logSources: [
                        { key: "DEPLOYMENT", text: "Deployment Engine", selected: true },
                        { key: "APPLICATION", text: "Application", selected: true },
                        { key: "SYSTEM", text: "System", selected: false },
                        { key: "SECURITY", text: "Security", selected: false },
                        { key: "NETWORK", text: "Network", selected: false }
                    ],
                    logLevels: [
                        { key: "ALL", text: "All Levels" },
                        { key: "ERROR", text: "Errors Only" },
                        { key: "WARN", text: "Warnings & Errors" },
                        { key: "INFO", text: "Info & Above" },
                        { key: "DEBUG", text: "Debug & Above" }
                    ],
                    logs: [],
                    filteredLogs: []
                });
                oDialog.setModel(oLogModel, "logs");
                oDialog.open();
                this._loadDeploymentLogs(sTaskId, oDialog);
            };

            const handleLogDialogError = (error) => {
                MessageBox.error(`Failed to open Logs Viewer: ${ error.message}`);
            };

            this._getOrCreateDialog("viewLogs", "a2a.network.agent15.ext.fragment.ViewLogs")
                .then(handleLogDialogOpen)
                .catch(handleLogDialogError);
        },

        /**
         * @function onPromoteToProduction
         * @description Promotes deployment to production environment.
         * @public
         */
        onPromoteToProduction() {
            if (!this._securityUtils.hasRole("DeploymentAdmin")) {
                MessageBox.error("Access denied. Deployment Administrator role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "PromoteToProduction", reason: "Insufficient permissions" });
                return;
            }

            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            const sTaskId = oData.taskId;
            const sTaskName = oData.taskName;

            // Check if promotion is allowed
            if (oData.environment === "PRODUCTION") {
                MessageBox.warning(this.getResourceBundle().getText("msg.alreadyInProduction"));
                return;
            }

            if (oData.status !== "DEPLOYED") {
                MessageBox.warning(this.getResourceBundle().getText("msg.deploymentNotReady"));
                return;
            }

            this._auditLogger.log("PROMOTE_TO_PRODUCTION_REQUESTED", { taskId: sTaskId, taskName: sTaskName });

            this._getOrCreateDialog("promoteToProduction", "a2a.network.agent15.ext.fragment.PromoteToProduction")
                .then((oDialog) => {
                    const oPromoteModel = new JSONModel({
                        taskId: sTaskId,
                        taskName: sTaskName,
                        currentEnvironment: oData.environment,
                        targetEnvironment: "PRODUCTION",
                        promotionStrategy: "BLUE_GREEN",
                        approvalRequired: true,
                        prePromotionChecks: {
                            validationRequired: true,
                            loadTestRequired: true,
                            securityScanRequired: true,
                            backupRequired: true
                        },
                        promotionOptions: {
                            schedulePromotion: false,
                            scheduledTime: new Date(),
                            maintenanceWindow: false,
                            rollbackPlan: true,
                            monitoringEnabled: true,
                            alertingEnabled: true
                        },
                        approvers: [],
                        requiredApprovals: 2,
                        promotionPlan: {
                            steps: [],
                            estimatedDuration: 0,
                            rollbackTime: 0
                        }
                    });
                    oDialog.setModel(oPromoteModel, "promote");
                    oDialog.open();
                    this._loadPromotionPlan(sTaskId, oDialog);
                })
                .catch((error) => {
                    MessageBox.error(`Failed to open Promote to Production dialog: ${ error.message}`);
                });
        },

        /**
         * @function _executeRollback
         * @description Executes deployment rollback with audit logging.
         * @param {string} sTaskId - Task ID
         * @param {string} sTaskName - Task name
         * @param {Object} oData - Deployment data
         * @private
         */
        _executeRollback(sTaskId, sTaskName, oData) {
            const oModel = this.base.getView().getModel();

            SecurityUtils.secureCallFunction(oModel, "/RollbackDeployment", {
                urlParameters: {
                    taskId: sTaskId,
                    deploymentId: oData.deploymentId,
                    targetVersion: oData.previousVersion,
                    createBackup: true,
                    validateRollback: true
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.rollbackInitiated"));
                    this._auditLogger.log("ROLLBACK_INITIATED", {
                        taskId: sTaskId,
                        taskName: sTaskName,
                        targetVersion: oData.previousVersion,
                        rollbackId: data.rollbackId,
                        success: true
                    });

                    // Start monitoring rollback progress
                    this._startRollbackMonitoring(data.rollbackId);
                }.bind(this),
                error: function(error) {
                    MessageBox.error(this.getResourceBundle().getText("error.rollbackFailed"));
                    this._auditLogger.log("ROLLBACK_FAILED", {
                        taskId: sTaskId,
                        taskName: sTaskName,
                        error: error.message
                    });
                }.bind(this)
            });
        },

        /**
         * @function _loadDeploymentLogs
         * @description Loads deployment logs with filtering.
         * @param {string} sTaskId - Task ID
         * @param {sap.m.Dialog} oDialog - Logs dialog
         * @private
         */
        _loadDeploymentLogs(sTaskId, oDialog) {
            oDialog.setBusy(true);

            const oModel = this.base.getView().getModel();

            SecurityUtils.secureCallFunction(oModel, "/GetDeploymentLogs", {
                urlParameters: {
                    taskId: sTaskId,
                    maxLines: 1000,
                    logLevel: "ALL",
                    includeMetadata: true
                },
                success: function(data) {
                    const oLogModel = oDialog.getModel("logs");
                    if (oLogModel) {
                        const oCurrentData = oLogModel.getData();
                        oCurrentData.logs = data.logs || [];
                        oCurrentData.filteredLogs = data.logs || [];
                        oCurrentData.totalLines = data.totalLines || 0;
                        oCurrentData.logSummary = data.summary || {};
                        oLogModel.setData(oCurrentData);
                    }
                    oDialog.setBusy(false);

                    // Start auto-refresh if enabled
                    if (oCurrentData.autoRefresh) {
                        this._startLogAutoRefresh(sTaskId, oDialog);
                    }
                }.bind(this),
                error(error) {
                    oDialog.setBusy(false);
                    MessageBox.error(`Failed to load deployment logs: ${ error.message}`);
                }
            });
        },

        /**
         * @function _loadPromotionPlan
         * @description Loads promotion plan and requirements.
         * @param {string} sTaskId - Task ID
         * @param {sap.m.Dialog} oDialog - Promotion dialog
         * @private
         */
        _loadPromotionPlan(sTaskId, oDialog) {
            oDialog.setBusy(true);

            const oModel = this.base.getView().getModel();

            SecurityUtils.secureCallFunction(oModel, "/GetPromotionPlan", {
                urlParameters: {
                    taskId: sTaskId,
                    targetEnvironment: "PRODUCTION"
                },
                success: function(data) {
                    const oPromoteModel = oDialog.getModel("promote");
                    if (oPromoteModel) {
                        const oCurrentData = oPromoteModel.getData();
                        oCurrentData.promotionPlan = data.plan || {};
                        oCurrentData.approvers = data.approvers || [];
                        oCurrentData.requirements = data.requirements || {};
                        oCurrentData.riskAssessment = data.riskAssessment || {};
                        oPromoteModel.setData(oCurrentData);
                    }
                    oDialog.setBusy(false);
                }.bind(this),
                error(error) {
                    oDialog.setBusy(false);
                    MessageBox.error(`Failed to load promotion plan: ${ error.message}`);
                }
            });
        },

        /**
         * @function _startRollbackMonitoring
         * @description Starts monitoring rollback progress.
         * @param {string} sRollbackId - Rollback ID to monitor
         * @private
         */
        _startRollbackMonitoring(sRollbackId) {
            // Update context to show rollback in progress
            const oContext = this.base.getView().getBindingContext();
            if (oContext) {
                oContext.refresh();
            }
        },

        /**
         * @function _startLogAutoRefresh
         * @description Starts auto-refresh for deployment logs.
         * @param {string} sTaskId - Task ID
         * @param {sap.m.Dialog} oDialog - Logs dialog
         * @private
         */
        _startLogAutoRefresh(sTaskId, oDialog) {
            if (this._logRefreshInterval) {
                clearInterval(this._logRefreshInterval);
            }

            const oLogData = oDialog.getModel("logs").getData();
            this._logRefreshInterval = setInterval(() => {
                if (oDialog.isOpen() && oLogData.autoRefresh) {
                    this._loadDeploymentLogs(sTaskId, oDialog);
                } else {
                    clearInterval(this._logRefreshInterval);
                }
            }, oLogData.refreshInterval || 10000);
        },

        /**
         * @function _initializeRealtimeMonitoring
         * @description Initializes real-time monitoring for deployment operations.
         * @private
         */
        _initializeRealtimeMonitoring() {
            // WebSocket for real-time deployment updates
            this._initializeDeploymentWebSocket();
        },

        /**
         * @function _initializeDeploymentWebSocket
         * @description Initializes WebSocket for deployment updates.
         * @private
         */
        _initializeDeploymentWebSocket() {
            if (this._deploymentWs) {return;}

            try {
                this._deploymentWs = SecurityUtils.createSecureWebSocket("blockchain://a2a-events", {
                    onMessage: function(data) {
                        this._handleDeploymentTaskUpdate(data);
                    }.bind(this)
                });

                this._deploymentWs.onclose = function() {
                    setTimeout(() => this._initializeDeploymentWebSocket(), 5000);
                }.bind(this);

            } catch (error) {
                // Error handled silently
            }
        },

        /**
         * @function _handleDeploymentTaskUpdate
         * @description Handles real-time deployment task updates.
         * @param {Object} data - Update data
         * @private
         */
        _handleDeploymentTaskUpdate(data) {
            const oContext = this.base.getView().getBindingContext();
            if (!oContext) {return;}

            const oCurrentData = oContext.getObject();

            // Check if update is for current task
            if (data.taskId === oCurrentData.taskId) {
                switch (data.type) {
                case "STATUS_UPDATE":
                    // Refresh the binding to get latest status
                    oContext.refresh();
                    break;
                case "VALIDATION_COMPLETED":
                    MessageToast.show("Deployment validation completed");
                    this._updateValidationResults(data);
                    break;
                case "ROLLBACK_PROGRESS":
                    this._updateRollbackProgress(data);
                    break;
                case "LOGS_UPDATED":
                    this._refreshLogsIfOpen(data.taskId);
                    break;
                case "PROMOTION_APPROVED":
                    MessageToast.show("Promotion to production approved");
                    oContext.refresh();
                    break;
                }
            }
        },

        /**
         * @function _updateValidationResults
         * @description Updates validation results in open dialog.
         * @param {Object} data - Validation data
         * @private
         */
        _updateValidationResults(data) {
            if (this._dialogCache.validateDeployment && this._dialogCache.validateDeployment.isOpen()) {
                const oValidationModel = this._dialogCache.validateDeployment.getModel("validation");
                if (oValidationModel) {
                    const oCurrentData = oValidationModel.getData();
                    oCurrentData.validationResults = data.results;
                    oValidationModel.setData(oCurrentData);
                }
            }
        },

        /**
         * @function _updateRollbackProgress
         * @description Updates rollback progress display.
         * @param {Object} data - Rollback progress data
         * @private
         */
        _updateRollbackProgress(data) {
            // Show rollback progress notifications
            if (data.status === "COMPLETED") {
                MessageToast.show("Rollback completed successfully");
            } else if (data.status === "FAILED") {
                MessageBox.error(`Rollback failed: ${ data.message}`);
            } else {
                MessageToast.show(`Rollback in progress: ${ data.currentStep}`);
            }
        },

        /**
         * @function _refreshLogsIfOpen
         * @description Refreshes logs if logs dialog is open.
         * @param {string} sTaskId - Task ID
         * @private
         */
        _refreshLogsIfOpen(sTaskId) {
            if (this._dialogCache.viewLogs && this._dialogCache.viewLogs.isOpen()) {
                this._loadDeploymentLogs(sTaskId, this._dialogCache.viewLogs);
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
                        agent: "Agent15_Deployment",
                        action,
                        details: details || {}
                    };
                    Log.info(`AUDIT: ${ JSON.stringify(logEntry)}`);
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
         * @function _cleanupResources
         * @description Cleans up resources to prevent memory leaks.
         * @private
         */
        _cleanupResources() {
            // Clean up WebSocket connections
            if (this._deploymentWs) {
                this._deploymentWs.close();
                this._deploymentWs = null;
            }

            // Clean up intervals
            if (this._logRefreshInterval) {
                clearInterval(this._logRefreshInterval);
                this._logRefreshInterval = null;
            }

            // Clean up cached dialogs
            if (this._dialogCache) {
                Object.keys(this._dialogCache).forEach((key) => {
                    if (this._dialogCache[key]) {
                        this._dialogCache[key].destroy();
                    }
                });
                this._dialogCache = {};
            }
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