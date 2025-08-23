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
     * @class a2a.network.agent6.ext.controller.ListReportExt
     * @extends sap.ui.core.mvc.ControllerExtension
     * @description Controller extension for Agent 6 List Report - Quality Control and Orchestration functionality.
     * Provides quality assessment, routing decision management, trust verification, and workflow optimization features.
     */
    return ControllerExtension.extend("a2a.network.agent6.ext.controller.ListReportExt", {
        
        override: {
            /**
             * @function onInit
             * @description Initializes the controller extension with security features and device model.
             * @override
             */
            onInit: function () {
                this._extensionAPI = this.base.getExtensionAPI();
                this._securityUtils = SecurityUtils;
                this._resourceBundle = this.base.getView().getModel("i18n").getResourceBundle();
                this._initializeSecurity();
                this._initializeDeviceModel();
                this._initializeDialogCache();
            },
            
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
         * @description Initializes dialog cache for performance optimization.
         * @private
         */
        _initializeDialogCache: function() {
            this._dialogCache = {};
        },

        /**
         * @function _initializeSecurity
         * @description Initializes security features including audit logging.
         * @private
         */
        _initializeSecurity: function() {
            this._auditLogger = {
                log: function(action, details) {
                    const user = this._getCurrentUser();
                    const timestamp = new Date().toISOString();
                    const logEntry = {
                        timestamp: timestamp,
                        user: user,
                        agent: "Agent6_QualityControl",
                        action: action,
                        details: details
                    };
                    Log.info("AUDIT: " + JSON.stringify(logEntry));
                }.bind(this)
            };
        },
        
        /**
         * @function _getQualityThresholds
         * @description Gets quality thresholds from configuration
         * @returns {Object} Quality thresholds configuration
         * @private
         */
        _getQualityThresholds: function() {
            // Get from model configuration or default to secure values
            var oConfigModel = this.base.getView().getModel("config");
            if (oConfigModel) {
                var thresholds = oConfigModel.getProperty("/qualityThresholds");
                if (thresholds && this._securityUtils.validateQualityThreshold(thresholds)) {
                    return thresholds;
                }
            }
            
            // Return secure defaults if no valid configuration
            return {
                minQualityScore: 80,
                maxIssues: 5,
                minTrustScore: 75,
                maxDefects: 10,
                maxWarnings: 20
            };
        },
        
        /**
         * @function _getCurrentUser
         * @description Retrieves current user ID from shell container.
         * @returns {string} User ID or 'anonymous'
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
            return user && user.hasRole && user.hasRole(role);
        },
        
        /**
         * @function _validateInput
         * @description Validates user input based on type with security patterns.
         * @param {string} input - Input to validate
         * @param {string} type - Type of validation
         * @returns {boolean} True if input is valid
         * @private
         */
        _validateInput: function(input, type) {
            if (!input || typeof input !== 'string') return false;
            
            switch(type) {
                case 'taskName':
                    return /^[a-zA-Z0-9\s\-_]{1,100}$/.test(input);
                case 'description':
                    return input.length <= 1000;
                case 'qualityGate':
                    return /^[A-Z0-9_]{1,50}$/.test(input);
                default:
                    return input.length > 0 && input.length <= 255;
            }
        },
        
        /**
         * @function _sanitizeInput
         * @description Sanitizes input to prevent XSS attacks.
         * @param {*} input - Input to sanitize
         * @returns {string} Sanitized input
         * @private
         */
        _sanitizeInput: function(input) {
            if (!input) return "";
            return encodeXML(input.toString().trim());
        },
        
        /**
         * @function _getCsrfToken
         * @description Retrieves CSRF token for secure requests.
         * @returns {Promise<string>} Promise resolving to CSRF token
         * @private
         */
        _getCsrfToken: function() {
            return new Promise((resolve, reject) => {
                jQuery.ajax({
                    url: "/a2a/agent6/v1/csrf-token",
                    type: "GET",
                    headers: {
                        "X-CSRF-Token": "Fetch"
                    },
                    success: function(data, textStatus, xhr) {
                        const token = xhr.getResponseHeader("X-CSRF-Token");
                        resolve(token);
                    },
                    error: function(xhr) {
                        reject(new Error("Failed to fetch CSRF token"));
                    }
                });
            });
        },
        
        /**
         * @function _secureAjaxCall
         * @description Makes secure AJAX call with CSRF token and error recovery.
         * @param {Object} options - jQuery AJAX options
         * @returns {Promise} Promise for the AJAX request
         * @private
         */
        _secureAjaxCall: function(options) {
            return this._getCsrfToken().then(token => {
                return new Promise((resolve, reject) => {
                    const secureOptions = Object.assign({}, options, {
                        headers: Object.assign({
                            "X-CSRF-Token": token,
                            "Content-Type": "application/json"
                        }, options.headers || {}),
                        success: function(data, textStatus, xhr) {
                            resolve({ data, textStatus, xhr });
                        },
                        error: function(xhr, textStatus, errorThrown) {
                            reject({ xhr, textStatus, errorThrown });
                        }
                    });
                    
                    jQuery.ajax(secureOptions);
                });
            });
        },
        
        /**
         * @function _cleanupResources
         * @description Cleans up dialog resources to prevent memory leaks.
         * @private
         */
        _cleanupResources: function() {
            // Clean up cached dialogs
            Object.keys(this._dialogCache).forEach(function(key) {
                if (this._dialogCache[key]) {
                    this._dialogCache[key].destroy();
                }
            }.bind(this));
            this._dialogCache = {};
            
            // Clean up legacy dialog references for backward compatibility
            if (this._oDashboard) {
                this._oDashboard.destroy();
                this._oDashboard = null;
            }
            if (this._oCreateDialog) {
                this._oCreateDialog.destroy();
                this._oCreateDialog = null;
            }
            if (this._oRoutingDialog) {
                this._oRoutingDialog.destroy();
                this._oRoutingDialog = null;
            }
            if (this._oTrustDialog) {
                this._oTrustDialog.destroy();
                this._oTrustDialog = null;
            }
            if (this._oWorkflowDialog) {
                this._oWorkflowDialog.destroy();
                this._oWorkflowDialog = null;
            }
        },

        /**
         * @function _getOrCreateDialog
         * @description Gets cached dialog or creates new one for performance.
         * @param {string} sDialogId - Dialog identifier
         * @param {string} sFragmentName - Fragment name to load
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
            // Add keyboard navigation
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
         * @description Optimizes dialog for current device type.
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
            
            // Add resize handler
            sap.ui.Device.resize.attachHandler(function() {
                if (sap.ui.Device.system.phone) {
                    oDialog.setStretch(true);
                } else {
                    oDialog.setStretch(false);
                }
            });
        },

        /**
         * @function _withErrorRecovery
         * @description Wraps operation with error recovery and retry logic.
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
         * @function onCreateQualityTask
         * @description Opens dialog to create new quality control task.
         * @public
         */
        onCreateQualityTask: function() {
            if (!this._hasRole("QualityManager")) {
                MessageBox.error("Access denied: Insufficient privileges for creating quality tasks");
                this._auditLogger.log("CREATE_TASK_ACCESS_DENIED", { action: "create_quality_task" });
                return;
            }
            
            this._auditLogger.log("CREATE_TASK_INITIATED", { action: "create_quality_task" });
            var oView = this.base.getView();
            
            this._getOrCreateDialog("create", "a2a.network.agent6.ext.fragment.CreateQualityTask")
                .then(function(oDialog) {
                    
                    var oModel = new JSONModel({
                        taskName: "",
                        description: "",
                        qualityGate: "",
                        dataSource: "",
                        processingPipeline: "",
                        evaluationCriteria: {
                            compliance: true,
                            performance: true,
                            security: true,
                            reliability: true,
                            usability: false,
                            maintainability: false
                        },
                        thresholds: this._getQualityThresholds()
                    });
                    oDialog.setModel(oModel, "create");
                    oDialog.open();
                }.bind(this));
        },

        /**
         * @function onQualityDashboard
         * @description Opens quality control dashboard with metrics and visualizations.
         * @public
         */
        onQualityDashboard: function() {
            var oView = this.base.getView();
            
            this._getOrCreateDialog("dashboard", "a2a.network.agent6.ext.fragment.QualityDashboard")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadDashboardData();
                }.bind(this));
        },

        _loadDashboardData: function() {
            this._oDashboard.setBusy(true);
            
            this._secureAjaxCall({
                url: "/a2a/agent6/v1/dashboard",
                type: "GET"
            }).then(result => {
                oDialog.setBusy(false);
                
                const sanitizedData = {
                    metrics: this._sanitizeObject(result.data.metrics),
                    trends: this._sanitizeArray(result.data.trends),
                    qualityGates: this._sanitizeArray(result.data.qualityGates),
                    routingStats: this._sanitizeObject(result.data.routingStats),
                    trustMetrics: this._sanitizeObject(result.data.trustMetrics),
                    workflowHealth: this._sanitizeObject(result.data.workflowHealth)
                };
                
                var oDashboardModel = new JSONModel(sanitizedData);
                this._oDashboard.setModel(oDashboardModel, "dashboard");
                this._createDashboardCharts(sanitizedData);
                
                this._auditLogger.log("DASHBOARD_LOADED", { action: "load_dashboard" });
            }).catch(error => {
                this._oDashboard.setBusy(false);
                const errorMsg = this._sanitizeInput(error.xhr?.responseText || "Unknown error");
                MessageBox.error(this._resourceBundle.getText("error.loadingDashboard", [errorMsg]));
                this._auditLogger.log("DASHBOARD_LOAD_FAILED", { error: errorMsg });
            });
        },

        _createDashboardCharts: function(data) {
            this._createQualityTrendsChart(data.trends);
            this._createRoutingDistributionChart(data.routingStats);
            this._createTrustScoreDistribution(data.trustMetrics);
        },

        _createQualityTrendsChart: function(trends) {
            var oVizFrame = this._oDashboard.byId("qualityTrendsChart");
            if (!oVizFrame || !trends) return;
            
            var aChartData = trends.map(function(trend) {
                return {
                    Date: trend.date,
                    QualityScore: trend.averageQuality,
                    TaskCount: trend.taskCount,
                    IssueCount: trend.issueCount
                };
            });
            
            var oChartModel = new sap.ui.model.json.JSONModel({
                trendData: aChartData
            });
            oVizFrame.setModel(oChartModel);
            
            oVizFrame.setVizProperties({
                categoryAxis: {
                    title: { text: "Date" }
                },
                valueAxis: {
                    title: { text: "Quality Score" }
                },
                title: {
                    text: "Quality Trends Over Time"
                }
            });
        },

        _createRoutingDistributionChart: function(routingStats) {
            var oVizFrame = this._oDashboard.byId("routingDistributionChart");
            if (!oVizFrame || !routingStats) return;
            
            var aChartData = Object.keys(routingStats).map(function(agent) {
                return {
                    Agent: agent,
                    TaskCount: routingStats[agent].count,
                    SuccessRate: routingStats[agent].successRate
                };
            });
            
            var oChartModel = new sap.ui.model.json.JSONModel({
                routingData: aChartData
            });
            oVizFrame.setModel(oChartModel);
        },

        /**
         * @function onRoutingDecisionManager
         * @description Opens routing decision management interface.
         * @public
         */
        onRoutingDecisionManager: function() {
            var oView = this.base.getView();
            
            this._getOrCreateDialog("routing", "a2a.network.agent6.ext.fragment.RoutingDecisionManager")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadRoutingData();
                }.bind(this));
        },

        _loadRoutingData: function() {
            this._secureAjaxCall({
                url: "/a2a/agent6/v1/routing-rules",
                type: "GET"
            }).then(result => {
                const sanitizedData = {
                    rules: this._sanitizeArray(result.data.rules),
                    agents: this._sanitizeArray(result.data.availableAgents),
                    pendingDecisions: this._sanitizeArray(result.data.pendingDecisions)
                };
                
                var oModel = new JSONModel(sanitizedData);
                this._oRoutingDialog.setModel(oModel, "routing");
                
                this._auditLogger.log("ROUTING_DATA_LOADED", { action: "load_routing_data" });
            }).catch(error => {
                const errorMsg = this._sanitizeInput(error.xhr?.responseText || "Unknown error");
                MessageBox.error(this._resourceBundle.getText("error.loadingRoutingData", [errorMsg]));
                this._auditLogger.log("ROUTING_DATA_LOAD_FAILED", { error: errorMsg });
            });
        },

        /**
         * @function onTrustVerification
         * @description Opens trust verification interface for blockchain integration.
         * @public
         */
        onTrustVerification: function() {
            var oView = this.base.getView();
            
            this._getOrCreateDialog("trust", "a2a.network.agent6.ext.fragment.TrustVerification")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadTrustData();
                }.bind(this));
        },

        _loadTrustData: function() {
            this._secureAjaxCall({
                url: "/a2a/agent6/v1/trust-metrics",
                type: "GET"
            }).then(result => {
                const sanitizedData = {
                    verificationQueue: this._sanitizeArray(result.data.verificationQueue),
                    trustFactors: this._sanitizeArray(result.data.trustFactors),
                    blockchainStatus: this._sanitizeObject(result.data.blockchainStatus),
                    reputationScores: this._sanitizeArray(result.data.reputationScores)
                };
                
                var oModel = new JSONModel(sanitizedData);
                this._oTrustDialog.setModel(oModel, "trust");
                
                this._auditLogger.log("TRUST_DATA_LOADED", { action: "load_trust_data" });
            }).catch(error => {
                const errorMsg = this._sanitizeInput(error.xhr?.responseText || "Unknown error");
                MessageBox.error("Failed to load trust data: " + errorMsg);
                this._auditLogger.log("TRUST_DATA_LOAD_FAILED", { error: errorMsg });
            });
        },

        /**
         * @function onWorkflowOptimization
         * @description Opens workflow optimization analysis dialog.
         * @public
         */
        onWorkflowOptimization: function() {
            var oView = this.base.getView();
            
            this._getOrCreateDialog("workflow", "a2a.network.agent6.ext.fragment.WorkflowOptimization")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadWorkflowData();
                }.bind(this));
        },

        _loadWorkflowData: function() {
            this._secureAjaxCall({
                url: "/a2a/agent6/v1/workflow-analysis",
                type: "GET"
            }).then(result => {
                const sanitizedData = {
                    bottlenecks: this._sanitizeArray(result.data.bottlenecks),
                    optimizations: this._sanitizeArray(result.data.optimizations),
                    performance: this._sanitizeArray(result.data.performance),
                    resourceUtilization: this._sanitizeObject(result.data.resourceUtilization)
                };
                
                var oModel = new JSONModel(sanitizedData);
                this._oWorkflowDialog.setModel(oModel, "workflow");
                this._createWorkflowVisualizations(sanitizedData);
                
                this._auditLogger.log("WORKFLOW_DATA_LOADED", { action: "load_workflow_data" });
            }).catch(error => {
                const errorMsg = this._sanitizeInput(error.xhr?.responseText || "Unknown error");
                MessageBox.error("Failed to load workflow data: " + errorMsg);
                this._auditLogger.log("WORKFLOW_DATA_LOAD_FAILED", { error: errorMsg });
            });
        },

        _createWorkflowVisualizations: function(data) {
            var oChart = this._oWorkflowDialog.byId("workflowChart");
            if (!oChart) return;
            
            var aChartData = data.performance.map(function(perf) {
                return {
                    Stage: perf.stage,
                    Duration: perf.averageDuration,
                    Efficiency: perf.efficiency
                };
            });
            
            var oChartModel = new sap.ui.model.json.JSONModel({
                performanceData: aChartData
            });
            oChart.setModel(oChartModel);
        },

        /**
         * @function onBatchAssessment
         * @description Initiates batch quality assessment for selected tasks.
         * @public
         */
        onBatchAssessment: function() {
            var oTable = this._extensionAPI.getTable();
            var aSelectedContexts = oTable.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageBox.warning("Please select at least one task for batch assessment.");
                return;
            }
            
            MessageBox.confirm(
                "Start batch quality assessment for " + aSelectedContexts.length + " tasks?",
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._startBatchAssessment(aSelectedContexts);
                        }
                    }.bind(this)
                }
            );
        },

        _startBatchAssessment: function(aContexts) {
            var aTaskIds = aContexts.map(function(oContext) {
                return this._sanitizeInput(oContext.getProperty("ID"));
            }.bind(this));
            
            if (aTaskIds.length > 50) {
                MessageBox.error("Batch size limited to 50 tasks for security reasons");
                return;
            }
            
            this.base.getView().setBusy(true);
            
            const requestData = {
                taskIds: aTaskIds,
                assessmentType: "COMPREHENSIVE",
                parallel: true
            };
            
            this._secureAjaxCall({
                url: "/a2a/agent6/v1/batch-assessment",
                type: "POST",
                data: JSON.stringify(requestData)
            }).then(result => {
                this.base.getView().setBusy(false);
                const data = result.data;
                MessageBox.success(
                    "Batch assessment started!\\n" +
                    "Job ID: " + this._sanitizeInput(data.jobId) + "\\n" +
                    "Estimated time: " + this._sanitizeInput(data.estimatedTime) + " minutes"
                );
                this._extensionAPI.refresh();
                
                this._auditLogger.log("BATCH_ASSESSMENT_STARTED", {
                    taskCount: aTaskIds.length,
                    jobId: data.jobId
                });
            }).catch(error => {
                this.base.getView().setBusy(false);
                const errorMsg = this._sanitizeInput(error.xhr?.responseText || "Unknown error");
                MessageBox.error("Batch assessment failed: " + errorMsg);
                this._auditLogger.log("BATCH_ASSESSMENT_FAILED", { error: errorMsg });
            });
        },

        /**
         * @function onConfigureQualityGates
         * @description Navigates to quality gates configuration.
         * @public
         */
        onConfigureQualityGates: function() {
            var oRouter = sap.ui.core.UIComponent.getRouterFor(this.base.getView());
            oRouter.navTo("QualityGates");
        },

        onConfirmCreateTask: function() {
            var oModel = this._oCreateDialog.getModel("create");
            var oData = oModel.getData();
            
            if (!this._validateInput(oData.taskName, 'taskName')) {
                MessageBox.error("Invalid task name. Use only alphanumeric characters, spaces, hyphens, and underscores (max 100 chars)");
                return;
            }
            
            if (!this._validateInput(oData.description, 'description')) {
                MessageBox.error("Description too long (max 1000 characters)");
                return;
            }
            
            if (!this._validateInput(oData.qualityGate, 'qualityGate')) {
                MessageBox.error("Invalid quality gate format. Use uppercase letters, numbers, and underscores only (max 50 chars)");
                return;
            }
            
            const sanitizedData = {
                taskName: this._sanitizeInput(oData.taskName),
                description: this._sanitizeInput(oData.description),
                qualityGate: this._sanitizeInput(oData.qualityGate),
                dataSource: this._sanitizeInput(oData.dataSource),
                processingPipeline: this._sanitizeInput(oData.processingPipeline),
                evaluationCriteria: oData.evaluationCriteria,
                thresholds: oData.thresholds
            };
            
            this._oCreateDialog.setBusy(true);
            
            this._secureAjaxCall({
                url: "/a2a/agent6/v1/tasks",
                type: "POST",
                data: JSON.stringify(sanitizedData)
            }).then(result => {
                this._oCreateDialog.setBusy(false);
                this._oCreateDialog.close();
                MessageToast.show("Quality control task created successfully");
                this._extensionAPI.refresh();
                
                this._auditLogger.log("TASK_CREATED", {
                    taskName: sanitizedData.taskName,
                    qualityGate: sanitizedData.qualityGate
                });
            }).catch(error => {
                this._oCreateDialog.setBusy(false);
                const errorMsg = this._sanitizeInput(error.xhr?.responseText || "Unknown error");
                MessageBox.error("Failed to create task: " + errorMsg);
                this._auditLogger.log("TASK_CREATION_FAILED", { error: errorMsg });
            });
        },

        onCancelCreateTask: function() {
            this._oCreateDialog.close();
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
        
        /**
         * @function _sanitizeArray
         * @description Sanitizes array elements.
         * @param {Array} arr - Array to sanitize
         * @returns {Array} Sanitized array
         * @private
         */
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
        }
    });
});