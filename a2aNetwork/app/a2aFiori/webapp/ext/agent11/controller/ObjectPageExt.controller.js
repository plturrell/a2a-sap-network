sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel",
    "a2a/network/agent11/ext/utils/SecurityUtils"
], function (ControllerExtension, MessageBox, MessageToast, Fragment, JSONModel, SecurityUtils) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent11.ext.controller.ObjectPageExt", {
        
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
            },
            
            onExit: function() {
                this._cleanupResources();
                if (this.base.onExit) {
                    this.base.onExit.apply(this, arguments);
                }
            }
        },

        /**
         * @function onViewAudit
         * @description Opens audit trail viewer with comprehensive compliance audit history.
         * @public
         */
        onViewAudit: function() {
            if (!this._hasRole("ComplianceUser")) {
                MessageBox.error("Access denied. Compliance User role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "ViewAudit", reason: "Insufficient permissions" });
                return;
            }

            const oContext = this.base.getView().getBindingContext();
            const sTaskId = oContext.getObject().taskId;
            const sTaskName = oContext.getObject().taskName;
            
            this._auditLogger.log("VIEW_AUDIT", { taskId: sTaskId, taskName: sTaskName });
            
            this._getOrCreateDialog("viewAudit", "a2a.network.agent11.ext.fragment.ViewAudit")
                .then(function(oDialog) {
                    var oAuditModel = new JSONModel({
                        taskId: sTaskId,
                        taskName: sTaskName,
                        dateRange: "LAST_30_DAYS",
                        startDate: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000),
                        endDate: new Date(),
                        auditLevel: "ALL",
                        actionFilter: "",
                        userFilter: "",
                        autoRefresh: false,
                        refreshInterval: 30000
                    });
                    oDialog.setModel(oAuditModel, "audit");
                    oDialog.open();
                    this._loadAuditTrail(sTaskId, oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Audit Trail: " + error.message);
                });
        },

        /**
         * @function onExportComplianceReport
         * @description Exports comprehensive compliance report with detailed analysis and charts.
         * @public
         */
        onExportComplianceReport: function() {
            if (!this._hasRole("ComplianceUser")) {
                MessageBox.error("Access denied. Compliance User role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "ExportComplianceReport", reason: "Insufficient permissions" });
                return;
            }

            const oContext = this.base.getView().getBindingContext();
            const sTaskId = oContext.getObject().taskId;
            const sTaskName = oContext.getObject().taskName;
            
            this._auditLogger.log("EXPORT_COMPLIANCE_REPORT", { taskId: sTaskId, taskName: sTaskName });
            
            this._getOrCreateDialog("exportCompliance", "a2a.network.agent11.ext.fragment.ExportComplianceReport")
                .then(function(oDialog) {
                    var oExportModel = new JSONModel({
                        taskId: sTaskId,
                        taskName: sTaskName,
                        reportType: "FULL_COMPLIANCE",
                        format: "PDF",
                        includeExecutiveSummary: true,
                        includeDetailedFindings: true,
                        includeRecommendations: true,
                        includeAuditTrail: true,
                        includeCharts: true,
                        includeRiskMatrix: true,
                        includeActionPlan: true,
                        includeAppendices: true,
                        confidentialityLevel: "INTERNAL",
                        distributionList: [],
                        customSections: []
                    });
                    oDialog.setModel(oExportModel, "export");
                    oDialog.open();
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Export Compliance Report: " + error.message);
                });
        },

        /**
         * @function onScheduleCompliance
         * @description Opens compliance scheduling interface for automated compliance checks.
         * @public
         */
        onScheduleCompliance: function() {
            if (!this._hasRole("ComplianceAdmin")) {
                MessageBox.error("Access denied. Compliance Administrator role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "ScheduleCompliance", reason: "Insufficient permissions" });
                return;
            }

            const oContext = this.base.getView().getBindingContext();
            const sTaskId = oContext.getObject().taskId;
            const sTaskName = oContext.getObject().taskName;
            
            this._auditLogger.log("SCHEDULE_COMPLIANCE", { taskId: sTaskId, taskName: sTaskName });
            
            this._getOrCreateDialog("scheduleCompliance", "a2a.network.agent11.ext.fragment.ScheduleCompliance")
                .then(function(oDialog) {
                    var oScheduleModel = new JSONModel({
                        taskId: sTaskId,
                        taskName: sTaskName,
                        scheduleType: "RECURRING",
                        frequency: "WEEKLY",
                        startDate: new Date(),
                        endDate: null,
                        timeOfDay: "09:00",
                        timezone: "UTC",
                        workdaysOnly: true,
                        notifyOnCompletion: true,
                        notifyOnFailure: true,
                        escalateFailures: true,
                        retryAttempts: 3,
                        retryInterval: 60,
                        enabled: true,
                        description: "",
                        customCronExpression: "",
                        notificationRecipients: [],
                        escalationPolicy: []
                    });
                    oDialog.setModel(oScheduleModel, "schedule");
                    oDialog.open();
                    this._loadScheduleOptions(oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Schedule Compliance: " + error.message);
                });
        },

        /**
         * @function onReviewViolations
         * @description Opens violations review interface with detailed analysis and remediation options.
         * @public
         */
        onReviewViolations: function() {
            if (!this._hasRole("ComplianceUser")) {
                MessageBox.error("Access denied. Compliance User role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "ReviewViolations", reason: "Insufficient permissions" });
                return;
            }

            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            const sTaskId = oData.taskId;
            const sTaskName = oData.taskName;
            
            // Check if there are violations to review
            if (!oData.violationCount || oData.violationCount === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.noViolationsToReview"));
                return;
            }
            
            this._auditLogger.log("REVIEW_VIOLATIONS", { 
                taskId: sTaskId, 
                taskName: sTaskName, 
                violationCount: oData.violationCount 
            });
            
            this._getOrCreateDialog("reviewViolations", "a2a.network.agent11.ext.fragment.ReviewViolations")
                .then(function(oDialog) {
                    var oViolationsModel = new JSONModel({
                        taskId: sTaskId,
                        taskName: sTaskName,
                        violationCount: oData.violationCount,
                        severityFilter: "ALL",
                        statusFilter: "OPEN",
                        categoryFilter: "ALL",
                        sortBy: "SEVERITY",
                        sortOrder: "DESC",
                        groupBy: "CATEGORY",
                        showResolved: false,
                        autoRefresh: true,
                        refreshInterval: 60000
                    });
                    oDialog.setModel(oViolationsModel, "violations");
                    oDialog.open();
                    this._loadViolationsData(sTaskId, oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Review Violations: " + error.message);
                });
        },

        /**
         * @function _loadAuditTrail
         * @description Loads audit trail data for a specific compliance task.
         * @param {string} sTaskId - Task ID
         * @param {sap.m.Dialog} oDialog - Audit dialog
         * @private
         */
        _loadAuditTrail: function(sTaskId, oDialog) {
            oDialog.setBusy(true);
            
            const oModel = this.base.getView().getModel();
            
            SecurityUtils.secureCallFunction(oModel, "/GetComplianceAuditTrail", {
                urlParameters: {
                    taskId: sTaskId,
                    limit: 100,
                    includeSystemEvents: true
                },
                success: function(data) {
                    var oAuditModel = oDialog.getModel("audit");
                    if (oAuditModel) {
                        var oCurrentData = oAuditModel.getData();
                        oCurrentData.auditEntries = data.auditEntries || [];
                        oCurrentData.totalCount = data.totalCount || 0;
                        oCurrentData.summaryStats = data.summaryStats || {};
                        oCurrentData.lastUpdated = new Date().toISOString();
                        oAuditModel.setData(oCurrentData);
                    }
                    oDialog.setBusy(false);
                }.bind(this),
                error: function(error) {
                    oDialog.setBusy(false);
                    MessageBox.error("Failed to load audit trail: " + error.message);
                }
            });
        },

        /**
         * @function _loadScheduleOptions
         * @description Loads scheduling options and existing schedules.
         * @param {sap.m.Dialog} oDialog - Schedule dialog
         * @private
         */
        _loadScheduleOptions: function(oDialog) {
            const oModel = this.base.getView().getModel();
            const oScheduleData = oDialog.getModel("schedule").getData();
            
            SecurityUtils.secureCallFunction(oModel, "/GetScheduleOptions", {
                urlParameters: { taskId: oScheduleData.taskId },
                success: function(data) {
                    var oScheduleModel = oDialog.getModel("schedule");
                    if (oScheduleModel) {
                        var oCurrentData = oScheduleModel.getData();
                        oCurrentData.availableTimezones = data.timezones || [];
                        oCurrentData.notificationChannels = data.notificationChannels || [];
                        oCurrentData.escalationPolicies = data.escalationPolicies || [];
                        oCurrentData.existingSchedules = data.existingSchedules || [];
                        oScheduleModel.setData(oCurrentData);
                    }
                }.bind(this),
                error: function(error) {
                    MessageBox.error("Failed to load schedule options: " + error.message);
                }
            });
        },

        /**
         * @function _loadViolationsData
         * @description Loads violations data with detailed analysis.
         * @param {string} sTaskId - Task ID
         * @param {sap.m.Dialog} oDialog - Violations dialog
         * @private
         */
        _loadViolationsData: function(sTaskId, oDialog) {
            oDialog.setBusy(true);
            
            const oModel = this.base.getView().getModel();
            
            SecurityUtils.secureCallFunction(oModel, "/GetComplianceViolations", {
                urlParameters: {
                    taskId: sTaskId,
                    includeDetails: true,
                    includeRecommendations: true
                },
                success: function(data) {
                    var oViolationsModel = oDialog.getModel("violations");
                    if (oViolationsModel) {
                        var oCurrentData = oViolationsModel.getData();
                        oCurrentData.violations = data.violations || [];
                        oCurrentData.summaryStats = data.summaryStats || {};
                        oCurrentData.riskMatrix = data.riskMatrix || {};
                        oCurrentData.trendAnalysis = data.trendAnalysis || {};
                        oCurrentData.remediationSuggestions = data.remediationSuggestions || [];
                        oCurrentData.lastUpdated = new Date().toISOString();
                        oViolationsModel.setData(oCurrentData);
                    }
                    oDialog.setBusy(false);
                    
                    // Start auto-refresh if enabled
                    if (oCurrentData.autoRefresh) {
                        this._startViolationsAutoRefresh(sTaskId, oDialog);
                    }
                }.bind(this),
                error: function(error) {
                    oDialog.setBusy(false);
                    MessageBox.error("Failed to load violations data: " + error.message);
                }
            });
        },

        /**
         * @function _startViolationsAutoRefresh
         * @description Starts auto-refresh for violations data.
         * @param {string} sTaskId - Task ID
         * @param {sap.m.Dialog} oDialog - Violations dialog
         * @private
         */
        _startViolationsAutoRefresh: function(sTaskId, oDialog) {
            if (this._violationsRefreshInterval) {
                clearInterval(this._violationsRefreshInterval);
            }
            
            var oViolationsData = oDialog.getModel("violations").getData();
            this._violationsRefreshInterval = setInterval(() => {
                if (oDialog.isOpen()) {
                    this._loadViolationsData(sTaskId, oDialog);
                } else {
                    clearInterval(this._violationsRefreshInterval);
                }
            }, oViolationsData.refreshInterval || 60000);
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
                        agent: "Agent11_Compliance",
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
            const mockRoles = ["ComplianceAdmin", "ComplianceUser", "ComplianceOperator"];
            return mockRoles.includes(role);
        },

        /**
         * @function _cleanupResources
         * @description Cleans up resources to prevent memory leaks.
         * @private
         */
        _cleanupResources: function() {
            // Clean up intervals
            if (this._violationsRefreshInterval) {
                clearInterval(this._violationsRefreshInterval);
                this._violationsRefreshInterval = null;
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