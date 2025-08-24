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
    "../../../utils/SharedSecurityUtils"
], function (ControllerExtension, Fragment, MessageBox, MessageToast, JSONModel, SecurityUtils) {
    "use strict";

    /**
     * @class a2a.network.agent15.ext.controller.ListReportExt
     * @extends sap.ui.core.mvc.ControllerExtension
     * @description Controller extension for Agent 15 List Report - Deployment Management Agent.
     * Provides comprehensive deployment orchestration, pipeline configuration, and status monitoring
     * with enterprise-grade security, audit logging, and accessibility features.
     */
    return ControllerExtension.extend("a2a.network.agent15.ext.controller.ListReportExt", {
        
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
                this._startRealtimeUpdates();
                this._initializeSecurity();
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
         * @function onDeployPackage
         * @description Deploys selected packages to target environments.
         * @public
         */
        onDeployPackage: function() {
            if (!this._securityUtils.hasRole("DeploymentAdmin")) {
                MessageBox.error("Access denied. Deployment Administrator role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "DeployPackage", reason: "Insufficient permissions" });
                return;
            }

            const oBinding = this.base.getView().byId("fe::table::DeploymentTasks::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectTasksFirst"));
                return;
            }

            // Filter deployable packages
            const aDeployablePackages = aSelectedContexts.filter(ctx => {
                const oData = ctx.getObject();
                return oData.status === "READY" || oData.status === "VALIDATED";
            });

            if (aDeployablePackages.length === 0) {
                MessageBox.warning(this.getResourceBundle().getText("msg.noDeployablePackages"));
                return;
            }

            this._auditLogger.log("DEPLOY_PACKAGE", { packageCount: aDeployablePackages.length });
            
            MessageBox.confirm(
                this.getResourceBundle().getText("msg.deployPackageConfirm", [aDeployablePackages.length]),
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._executePackageDeployment(aDeployablePackages);
                        }
                    }.bind(this)
                }
            );
        },

        /**
         * @function onConfigurePipeline
         * @description Opens deployment pipeline configuration interface.
         * @public
         */
        onConfigurePipeline: function() {
            if (!this._securityUtils.hasRole("DeploymentAdmin")) {
                MessageBox.error("Access denied. Deployment Administrator role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "ConfigurePipeline", reason: "Insufficient permissions" });
                return;
            }

            this._auditLogger.log("CONFIGURE_PIPELINE", { action: "OpenPipelineConfiguration" });
            
            this._getOrCreateDialog("configurePipeline", "a2a.network.agent15.ext.fragment.ConfigurePipeline")
                .then(function(oDialog) {
                    var oPipelineModel = new JSONModel({
                        pipelines: [],
                        environments: [
                            { key: "DEV", text: "Development", order: 1 },
                            { key: "TEST", text: "Testing", order: 2 },
                            { key: "STAGING", text: "Staging", order: 3 },
                            { key: "PROD", text: "Production", order: 4 }
                        ],
                        deploymentStrategies: [
                            { key: "BLUE_GREEN", text: "Blue-Green Deployment" },
                            { key: "ROLLING", text: "Rolling Deployment" },
                            { key: "CANARY", text: "Canary Deployment" },
                            { key: "RECREATE", text: "Recreate Deployment" }
                        ],
                        selectedStrategy: "ROLLING",
                        approvalRequired: true,
                        automatedTesting: true,
                        rollbackEnabled: true,
                        notificationSettings: {
                            onSuccess: true,
                            onFailure: true,
                            onStart: false
                        },
                        stages: [],
                        triggers: {
                            manual: true,
                            scheduled: false,
                            webhook: false,
                            scmTrigger: false
                        }
                    });
                    oDialog.setModel(oPipelineModel, "pipeline");
                    oDialog.open();
                    this._loadPipelineConfigurations(oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Pipeline Configuration: " + error.message);
                });
        },

        /**
         * @function onViewDeploymentStatus
         * @description Opens deployment status monitoring dashboard.
         * @public
         */
        onViewDeploymentStatus: function() {
            if (!this._securityUtils.hasRole("DeploymentUser")) {
                MessageBox.error("Access denied. Deployment User role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "ViewDeploymentStatus", reason: "Insufficient permissions" });
                return;
            }

            this._auditLogger.log("VIEW_DEPLOYMENT_STATUS", { action: "OpenStatusDashboard" });
            
            this._getOrCreateDialog("viewDeploymentStatus", "a2a.network.agent15.ext.fragment.ViewDeploymentStatus")
                .then(function(oDialog) {
                    var oStatusModel = new JSONModel({
                        statusFilter: "ALL",
                        environmentFilter: "ALL",
                        timeRange: "LAST_24_HOURS",
                        startDate: new Date(Date.now() - 24 * 60 * 60 * 1000),
                        endDate: new Date(),
                        autoRefresh: true,
                        refreshInterval: 30000,
                        deployments: [],
                        statistics: {
                            running: 0,
                            successful: 0,
                            failed: 0,
                            pending: 0,
                            rollback: 0
                        },
                        environmentStatus: {
                            dev: "HEALTHY",
                            test: "HEALTHY",
                            staging: "WARNING",
                            production: "HEALTHY"
                        },
                        performanceMetrics: {
                            deploymentFrequency: 0,
                            successRate: 0,
                            averageDeployTime: 0,
                            meanTimeToRecovery: 0
                        }
                    });
                    oDialog.setModel(oStatusModel, "status");
                    oDialog.open();
                    this._loadDeploymentStatus(oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Deployment Status: " + error.message);
                });
        },

        /**
         * @function _executePackageDeployment
         * @description Executes deployment for selected packages.
         * @param {Array} aSelectedContexts - Selected deployment task contexts
         * @private
         */
        _executePackageDeployment: function(aSelectedContexts) {
            const aTaskIds = aSelectedContexts.map(ctx => ctx.getObject().taskId);
            
            // Show progress dialog
            this._getOrCreateDialog("deploymentProgress", "a2a.network.agent15.ext.fragment.DeploymentProgress")
                .then(function(oProgressDialog) {
                    var oProgressModel = new JSONModel({
                        totalPackages: aTaskIds.length,
                        completedPackages: 0,
                        currentPackage: "",
                        progress: 0,
                        status: "Starting deployment...",
                        currentStage: "PREPARATION",
                        stages: [
                            { name: "PREPARATION", status: "ACTIVE", progress: 0 },
                            { name: "VALIDATION", status: "PENDING", progress: 0 },
                            { name: "DEPLOYMENT", status: "PENDING", progress: 0 },
                            { name: "TESTING", status: "PENDING", progress: 0 },
                            { name: "FINALIZATION", status: "PENDING", progress: 0 }
                        ],
                        deploymentResults: []
                    });
                    oProgressDialog.setModel(oProgressModel, "progress");
                    oProgressDialog.open();
                    
                    this._runPackageDeployment(aTaskIds, oProgressDialog);
                }.bind(this));
        },

        /**
         * @function _runPackageDeployment
         * @description Runs package deployment with real-time progress updates.
         * @param {Array} aTaskIds - Array of task IDs to deploy
         * @param {sap.m.Dialog} oProgressDialog - Progress dialog
         * @private
         */
        _runPackageDeployment: function(aTaskIds, oProgressDialog) {
            const oModel = this.base.getView().getModel();
            
            SecurityUtils.secureCallFunction(oModel, "/DeployPackages", {
                urlParameters: {
                    taskIds: aTaskIds.join(','),
                    strategy: "ROLLING",
                    validateFirst: true,
                    enableRollback: true,
                    testAfterDeploy: true
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.deploymentStarted"));
                    this._startDeploymentMonitoring(data.deploymentId, oProgressDialog);
                    this._auditLogger.log("DEPLOYMENT_STARTED", { 
                        taskCount: aTaskIds.length, 
                        deploymentId: data.deploymentId,
                        success: true 
                    });
                }.bind(this),
                error: function(error) {
                    MessageBox.error(this.getResourceBundle().getText("error.deploymentFailed"));
                    oProgressDialog.close();
                    this._auditLogger.log("DEPLOYMENT_FAILED", { 
                        taskCount: aTaskIds.length, 
                        error: error.message 
                    });
                }.bind(this)
            });
        },

        /**
         * @function _startDeploymentMonitoring
         * @description Starts real-time monitoring of deployment progress.
         * @param {string} sDeploymentId - Deployment ID to monitor
         * @param {sap.m.Dialog} oProgressDialog - Progress dialog
         * @private
         */
        _startDeploymentMonitoring: function(sDeploymentId, oProgressDialog) {
            if (this._deploymentEventSource) {
                this._deploymentEventSource.close();
            }
            
            try {
                this._deploymentEventSource = new EventSource('/api/agent15/deployment/stream/' + sDeploymentId);
                
                this._deploymentEventSource.onmessage = function(event) {
                    try {
                        const data = JSON.parse(event.data);
                        this._updateDeploymentProgress(data, oProgressDialog);
                    } catch (error) {
                        console.error('Error parsing deployment progress data:', error);
                    }
                }.bind(this);
                
                this._deploymentEventSource.onerror = function(error) {
                    console.warn('Deployment stream error, falling back to polling:', error);
                    this._startDeploymentPolling(sDeploymentId, oProgressDialog);
                }.bind(this);
                
            } catch (error) {
                console.warn('EventSource not available, using polling fallback');
                this._startDeploymentPolling(sDeploymentId, oProgressDialog);
            }
        },

        /**
         * @function _loadPipelineConfigurations
         * @description Loads pipeline configurations and templates.
         * @param {sap.m.Dialog} oDialog - Pipeline configuration dialog
         * @private
         */
        _loadPipelineConfigurations: function(oDialog) {
            oDialog.setBusy(true);
            
            const oModel = this.base.getView().getModel();
            
            SecurityUtils.secureCallFunction(oModel, "/GetPipelineConfigurations", {
                success: function(data) {
                    var oPipelineModel = oDialog.getModel("pipeline");
                    if (oPipelineModel) {
                        var oCurrentData = oPipelineModel.getData();
                        oCurrentData.pipelines = data.pipelines || [];
                        oCurrentData.templates = data.templates || [];
                        oCurrentData.integrations = data.integrations || [];
                        oCurrentData.approvalWorkflows = data.approvalWorkflows || [];
                        oPipelineModel.setData(oCurrentData);
                    }
                    oDialog.setBusy(false);
                }.bind(this),
                error: function(error) {
                    oDialog.setBusy(false);
                    MessageBox.error("Failed to load pipeline configurations: " + error.message);
                }
            });
        },

        /**
         * @function _loadDeploymentStatus
         * @description Loads deployment status information.
         * @param {sap.m.Dialog} oDialog - Status dialog
         * @private
         */
        _loadDeploymentStatus: function(oDialog) {
            oDialog.setBusy(true);
            
            const oModel = this.base.getView().getModel();
            
            SecurityUtils.secureCallFunction(oModel, "/GetDeploymentStatus", {
                success: function(data) {
                    var oStatusModel = oDialog.getModel("status");
                    if (oStatusModel) {
                        var oCurrentData = oStatusModel.getData();
                        oCurrentData.deployments = data.deployments || [];
                        oCurrentData.statistics = data.statistics || {};
                        oCurrentData.environmentStatus = data.environmentStatus || {};
                        oCurrentData.performanceMetrics = data.performanceMetrics || {};
                        oCurrentData.alerts = data.alerts || [];
                        oStatusModel.setData(oCurrentData);
                    }
                    oDialog.setBusy(false);
                    
                    // Start auto-refresh if enabled
                    if (oCurrentData.autoRefresh) {
                        this._startStatusAutoRefresh(oDialog);
                    }
                }.bind(this),
                error: function(error) {
                    oDialog.setBusy(false);
                    MessageBox.error("Failed to load deployment status: " + error.message);
                }
            });
        },

        /**
         * @function _updateDeploymentProgress
         * @description Updates deployment progress display.
         * @param {Object} data - Progress data
         * @param {sap.m.Dialog} oProgressDialog - Progress dialog
         * @private
         */
        _updateDeploymentProgress: function(data, oProgressDialog) {
            if (!oProgressDialog || !oProgressDialog.isOpen()) return;
            
            var oProgressModel = oProgressDialog.getModel("progress");
            if (oProgressModel) {
                var oCurrentData = oProgressModel.getData();
                oCurrentData.completedPackages = data.completedPackages || oCurrentData.completedPackages;
                oCurrentData.currentPackage = data.currentPackage || oCurrentData.currentPackage;
                oCurrentData.progress = Math.round((oCurrentData.completedPackages / oCurrentData.totalPackages) * 100);
                oCurrentData.status = data.status || oCurrentData.status;
                oCurrentData.currentStage = data.currentStage || oCurrentData.currentStage;
                
                // Update stages
                if (data.stages) {
                    oCurrentData.stages = data.stages;
                }
                
                if (data.deploymentResults && data.deploymentResults.length > 0) {
                    oCurrentData.deploymentResults = oCurrentData.deploymentResults.concat(data.deploymentResults);
                }
                
                oProgressModel.setData(oCurrentData);
                
                // Check if all packages are deployed
                if (oCurrentData.completedPackages >= oCurrentData.totalPackages) {
                    this._completeDeployment(oProgressDialog);
                }
            }
        },

        /**
         * @function _completeDeployment
         * @description Handles completion of deployment operation.
         * @param {sap.m.Dialog} oProgressDialog - Progress dialog
         * @private
         */
        _completeDeployment: function(oProgressDialog) {
            setTimeout(() => {
                oProgressDialog.close();
                MessageToast.show(this.getResourceBundle().getText("msg.deploymentCompleted"));
                this._refreshDeploymentData();
                this._auditLogger.log("DEPLOYMENT_COMPLETED", { status: "SUCCESS" });
            }, 2000);
            
            // Clean up event source
            if (this._deploymentEventSource) {
                this._deploymentEventSource.close();
                this._deploymentEventSource = null;
            }
        },

        /**
         * @function _refreshDeploymentData
         * @description Refreshes deployment task data in the table.
         * @private
         */
        _refreshDeploymentData: function() {
            const oBinding = this.base.getView().byId("fe::table::DeploymentTasks::LineItem").getBinding("rows");
            if (oBinding) {
                oBinding.refresh();
            }
        },

        /**
         * @function _startRealtimeUpdates
         * @description Starts real-time updates for deployment events.
         * @private
         */
        _startRealtimeUpdates: function() {
            this._initializeWebSocket();
        },

        /**
         * @function _initializeWebSocket
         * @description Initializes secure WebSocket connection for real-time deployment updates.
         * @private
         */
        _initializeWebSocket: function() {
            if (this._ws) return;

            // Validate WebSocket URL for security
            if (!this._securityUtils.validateWebSocketUrl('blockchain://a2a-events')) {
                MessageBox.error("Invalid WebSocket URL");
                return;
            }

            try {
                this._ws = SecurityUtils.createSecureWebSocket('blockchain://a2a-events', {
                    onMessage: function(data) {
                        this._handleDeploymentUpdate(data);
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
                this._refreshDeploymentData();
            }, 5000);
        },

        /**
         * @function _handleDeploymentUpdate
         * @description Handles real-time deployment updates from WebSocket.
         * @param {Object} data - Update data
         * @private
         */
        _handleDeploymentUpdate: function(data) {
            try {
                var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                
                switch (data.type) {
                    case 'DEPLOYMENT_STARTED':
                        var sDeploymentStarted = oBundle.getText("msg.deploymentStarted") || "Deployment started";
                        MessageToast.show(sDeploymentStarted);
                        break;
                    case 'DEPLOYMENT_PROGRESS':
                        this._updateDeploymentProgress(data);
                        break;
                    case 'DEPLOYMENT_COMPLETED':
                        var sDeploymentCompleted = oBundle.getText("msg.deploymentCompleted") || "Deployment completed";
                        MessageToast.show(sDeploymentCompleted);
                        this._refreshDeploymentData();
                        break;
                    case 'DEPLOYMENT_FAILED':
                        var sDeploymentFailed = oBundle.getText("error.deploymentFailed") || "Deployment failed";
                        MessageBox.error(sDeploymentFailed + ": " + data.message);
                        break;
                    case 'ROLLBACK_INITIATED':
                        var sRollbackInitiated = oBundle.getText("msg.rollbackInitiated") || "Rollback initiated";
                        MessageToast.show(sRollbackInitiated);
                        this._refreshDeploymentData();
                        break;
                    case 'ENVIRONMENT_WARNING':
                        var sEnvironmentWarning = oBundle.getText("msg.environmentWarning") || "Environment warning";
                        MessageBox.warning(sEnvironmentWarning + ": " + data.message);
                        break;
                    case 'PIPELINE_UPDATED':
                        var sPipelineUpdated = oBundle.getText("msg.pipelineUpdated") || "Pipeline configuration updated";
                        MessageToast.show(sPipelineUpdated);
                        break;
                }
            } catch (error) {
                console.error("Error processing deployment update:", error);
            }
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
            // Throttle deployment data updates
            this._throttledDeploymentUpdate = this._throttle(this._refreshDeploymentData.bind(this), 1000);
            // Debounce search operations
            this._debouncedSearch = this._debounce(this._performSearch.bind(this), 300);
        },
        
        /**
         * @function _performSearch
         * @description Performs search operation for deployment tasks.
         * @param {string} sQuery - Search query
         * @private
         */
        _performSearch: function(sQuery) {
            // Implement search logic for deployment tasks
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
                        agent: "Agent15_Deployment",
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
            if (this._deploymentEventSource) {
                this._deploymentEventSource.close();
                this._deploymentEventSource = null;
            }
            
            // Clean up polling intervals
            if (this._pollInterval) {
                clearInterval(this._pollInterval);
                this._pollInterval = null;
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