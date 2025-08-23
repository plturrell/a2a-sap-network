sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel",
    "../utils/SecurityUtils"
], function(ControllerExtension, MessageToast, MessageBox, Fragment, JSONModel, SecurityUtils) {
    "use strict";

    /**
     * @class a2a.network.agent13.ext.controller.ListReportExt
     * @extends sap.ui.core.mvc.ControllerExtension
     * @description Controller extension for Agent 13 List Report - Agent Builder Agent.
     * Provides comprehensive agent creation and deployment capabilities including template management,
     * code generation, pipeline orchestration, and deployment automation with enterprise-grade security.
     */
    return ControllerExtension.extend("a2a.network.agent13.ext.controller.ListReportExt", {
        
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
                this._startRealtimeBuilderUpdates();
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
            // Debounce template search operations
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
         * @description Performs search operation for agent templates.
         * @param {string} sQuery - Search query
         * @private
         */
        _performSearch: function(sQuery) {
            // Implement search logic for agent templates
        },

        /**
         * @function onBuilderDashboard
         * @description Opens comprehensive agent builder analytics dashboard with deployment metrics and pipeline status.
         * @public
         */
        onBuilderDashboard: function() {
            this._getOrCreateDialog("builderDashboard", "a2a.network.agent13.ext.fragment.BuilderDashboard")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadDashboardData(oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Builder Dashboard: " + error.message);
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
         * @function onCreateAgentTemplate
         * @description Opens template creation wizard for new agent template design.
         * @public
         */
        onCreateAgentTemplate: function() {
            this._getOrCreateDialog("templateWizard", "a2a.network.agent13.ext.fragment.TemplateWizard")
                .then(function(oDialog) {
                    var oModel = new JSONModel({
                        templateName: "",
                        description: "",
                        templateType: "basic",
                        agentCategory: "data_processing",
                        complexity: "moderate",
                        codeLanguage: "javascript",
                        frameworkVersion: "latest",
                        customizable: true,
                        autoGenerateTests: true,
                        autoGenerateDocs: true,
                        deploymentTarget: "kubernetes",
                        monitoringEnabled: true
                    });
                    oDialog.setModel(oModel, "create");
                    oDialog.open();
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Template Wizard: " + error.message);
                });
        },

        /**
         * @function onCodeGenerator
         * @description Opens code generation interface for selected agent template.
         * @public
         */
        onCodeGenerator: function() {
            const oBinding = this.base.getView().byId("fe::table::AgentTemplates::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectTemplatesFirst"));
                return;
            }

            this._getOrCreateDialog("codeGenerator", "a2a.network.agent13.ext.fragment.CodeGenerator")
                .then(function(oDialog) {
                    oDialog.open();
                    this._initializeCodeGenerator(aSelectedContexts[0], oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Code Generator: " + error.message);
                });
        },

        /**
         * @function onDeploymentManager
         * @description Opens deployment management interface for agent deployment orchestration.
         * @public
         */
        onDeploymentManager: function() {
            this._getOrCreateDialog("deploymentManager", "a2a.network.agent13.ext.fragment.DeploymentManager")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadDeploymentData(oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Deployment Manager: " + error.message);
                });
        },

        /**
         * @function onPipelineManager
         * @description Opens build pipeline management interface for CI/CD orchestration.
         * @public
         */
        onPipelineManager: function() {
            this._getOrCreateDialog("pipelineManager", "a2a.network.agent13.ext.fragment.PipelineManager")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadPipelineData(oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Pipeline Manager: " + error.message);
                });
        },

        /**
         * @function onComponentBuilder
         * @description Opens component builder interface for modular agent component design.
         * @public
         */
        onComponentBuilder: function() {
            const oBinding = this.base.getView().byId("fe::table::AgentTemplates::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectTemplatesFirst"));
                return;
            }

            this._getOrCreateDialog("componentBuilder", "a2a.network.agent13.ext.fragment.ComponentBuilder")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadComponentData(aSelectedContexts[0], oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Component Builder: " + error.message);
                });
        },

        /**
         * @function onTestHarness
         * @description Opens test harness interface for agent template testing and validation.
         * @public
         */
        onTestHarness: function() {
            const oBinding = this.base.getView().byId("fe::table::AgentTemplates::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectTemplatesFirst"));
                return;
            }

            this._getOrCreateDialog("testHarness", "a2a.network.agent13.ext.fragment.TestHarness")
                .then(function(oDialog) {
                    oDialog.open();
                    this._initializeTestHarness(aSelectedContexts[0], oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Test Harness: " + error.message);
                });
        },

        /**
         * @function onBatchBuild
         * @description Initiates batch build process for multiple agent templates.
         * @public
         */
        onBatchBuild: function() {
            const oBinding = this.base.getView().byId("fe::table::AgentTemplates::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectTemplatesFirst"));
                return;
            }

            MessageBox.confirm(
                this.getResourceBundle().getText("msg.batchBuildConfirm", [aSelectedContexts.length]),
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._startBatchBuild(aSelectedContexts);
                        }
                    }.bind(this)
                }
            );
        },

        /**
         * @function _startRealtimeBuilderUpdates
         * @description Starts real-time updates for build and deployment events.
         * @private
         */
        _startRealtimeBuilderUpdates: function() {
            this._initializeWebSocket();
        },

        /**
         * @function _initializeWebSocket
         * @description Initializes secure WebSocket connection for real-time builder updates.
         * @private
         */
        _initializeWebSocket: function() {
            if (this._ws) return;

            // Use secure WebSocket URL
            var wsUrl = 'wss://' + window.location.hostname + ':8013/builder/updates';
            
            try {
                this._ws = SecurityUtils.createSecureWebSocket(wsUrl, {
                    onmessage: function(event) {
                        try {
                            const data = JSON.parse(event.data);
                            this._handleBuilderUpdate(data);
                        } catch (error) {
                            console.error("Error parsing WebSocket message:", error);
                        }
                    }.bind(this),
                    onerror: function(error) {
                        console.warn("Secure WebSocket error:", error);
                        this._initializePolling();
                    }.bind(this)
                });
                
                if (this._ws) {
                    this._ws.onclose = function() {
                        var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                        var sMessage = oBundle.getText("msg.websocketDisconnected") || "Connection lost. Reconnecting...";
                        MessageToast.show(sMessage);
                        setTimeout(() => this._initializeWebSocket(), 5000);
                    }.bind(this);
                }

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
                this._refreshTemplateData();
            }, 10000);
        },

        /**
         * @function _handleBuilderUpdate
         * @description Handles real-time builder updates from WebSocket.
         * @param {Object} data - Update data
         * @private
         */
        _handleBuilderUpdate: function(data) {
            try {
                // Sanitize incoming data
                const sanitizedData = SecurityUtils.sanitizeBuilderData(JSON.stringify(data));
                const parsedData = JSON.parse(sanitizedData);
                
                var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                
                switch (parsedData.type) {
                    case 'BUILD_STARTED':
                        var sBuildStarted = oBundle.getText("msg.buildStarted") || "Build started";
                        MessageToast.show(sBuildStarted);
                        break;
                    case 'BUILD_COMPLETED':
                        var sBuildCompleted = oBundle.getText("msg.buildCompleted") || "Build completed";
                        MessageToast.show(sBuildCompleted);
                        this._refreshTemplateData();
                        break;
                    case 'BUILD_FAILED':
                        var sBuildFailed = oBundle.getText("error.buildFailed") || "Build failed";
                        MessageToast.show(sBuildFailed);
                        break;
                    case 'DEPLOYMENT_STARTED':
                        var sDeployStarted = oBundle.getText("msg.deploymentStarted") || "Deployment started";
                        MessageToast.show(sDeployStarted);
                        break;
                    case 'DEPLOYMENT_COMPLETED':
                        var sDeployCompleted = oBundle.getText("msg.deploymentCompleted") || "Deployment completed";
                        MessageToast.show(sDeployCompleted);
                        this._refreshTemplateData();
                        break;
                    case 'DEPLOYMENT_FAILED':
                        var sDeployFailed = oBundle.getText("error.deploymentFailed") || "Deployment failed";
                        MessageToast.show(sDeployFailed);
                        break;
                    case 'PIPELINE_UPDATE':
                        this._updatePipelineStatus(parsedData.pipeline);
                        break;
                    case 'TEST_COMPLETED':
                        this._refreshTemplateData();
                        break;
                }
            } catch (error) {
                console.error("Error processing builder update:", error);
            }
        },

        /**
         * @function _loadDashboardData
         * @description Loads builder dashboard data with statistics and deployment metrics.
         * @param {sap.m.Dialog} oDialog - Target dialog (optional)
         * @private
         */
        _loadDashboardData: function(oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["builderDashboard"];
            if (!oTargetDialog) return;
            
            oTargetDialog.setBusy(true);
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    var oModel = this.base.getView().getModel();
                    return SecurityUtils.secureCallFunction(oModel, "/GetBuilderStatistics", {
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
         * @function _initializeCodeGenerator
         * @description Initializes code generator with template details.
         * @param {sap.ui.model.Context} oContext - Selected template context
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _initializeCodeGenerator: function(oContext, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["codeGenerator"];
            if (!oTargetDialog) return;
            
            oTargetDialog.setBusy(true);
            const sTemplateId = oContext.getObject().templateId;
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    var oModel = this.base.getView().getModel();
                    return SecurityUtils.secureCallFunction(oModel, "/GetTemplateDetails", {
                        urlParameters: {
                            templateId: sTemplateId
                        },
                        success: function(data) {
                            resolve(data);
                        },
                        error: function(error) {
                            var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                            var sErrorMsg = oBundle.getText("error.loadingTemplateDetails") || "Error loading template details";
                            reject(new Error(sErrorMsg));
                        }.bind(this)
                    });
                }.bind(this));
            }.bind(this)).then(function(data) {
                oTargetDialog.setBusy(false);
                this._setupCodeGenerator(data, oTargetDialog);
            }.bind(this)).catch(function(error) {
                oTargetDialog.setBusy(false);
                MessageBox.error(error.message);
            });
        },

        /**
         * @function _loadDeploymentData
         * @description Loads deployment targets and configurations.
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _loadDeploymentData: function(oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["deploymentManager"];
            if (!oTargetDialog) return;
            
            oTargetDialog.setBusy(true);
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    var oModel = this.base.getView().getModel();
                    return SecurityUtils.secureCallFunction(oModel, "/GetDeploymentTargets", {
                        success: function(data) {
                            resolve(data);
                        },
                        error: function(error) {
                            var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                            var sErrorMsg = oBundle.getText("error.loadingDeploymentData") || "Error loading deployment data";
                            reject(new Error(sErrorMsg));
                        }.bind(this)
                    });
                }.bind(this));
            }.bind(this)).then(function(data) {
                oTargetDialog.setBusy(false);
                this._updateDeploymentTargets(data, oTargetDialog);
            }.bind(this)).catch(function(error) {
                oTargetDialog.setBusy(false);
                MessageBox.error(error.message);
            });
        },

        /**
         * @function _loadPipelineData
         * @description Loads build pipeline configurations and status.
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _loadPipelineData: function(oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["pipelineManager"];
            if (!oTargetDialog) return;
            
            oTargetDialog.setBusy(true);
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    var oModel = this.base.getView().getModel();
                    return SecurityUtils.secureCallFunction(oModel, "/GetBuildPipelines", {
                        success: function(data) {
                            resolve(data);
                        },
                        error: function(error) {
                            var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                            var sErrorMsg = oBundle.getText("error.loadingPipelineData") || "Error loading pipeline data";
                            reject(new Error(sErrorMsg));
                        }.bind(this)
                    });
                }.bind(this));
            }.bind(this)).then(function(data) {
                oTargetDialog.setBusy(false);
                this._updatePipelineList(data, oTargetDialog);
            }.bind(this)).catch(function(error) {
                oTargetDialog.setBusy(false);
                MessageBox.error(error.message);
            });
        },

        /**
         * @function _loadComponentData
         * @description Loads agent component data for selected template.
         * @param {sap.ui.model.Context} oContext - Selected template context
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _loadComponentData: function(oContext, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["componentBuilder"];
            if (!oTargetDialog) return;
            
            oTargetDialog.setBusy(true);
            const sTemplateId = oContext.getObject().templateId;
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    var oModel = this.base.getView().getModel();
                    return SecurityUtils.secureCallFunction(oModel, "/GetAgentComponents", {
                        urlParameters: {
                            templateId: sTemplateId
                        },
                        success: function(data) {
                            resolve(data);
                        },
                        error: function(error) {
                            var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                            var sErrorMsg = oBundle.getText("error.loadingComponentData") || "Error loading component data";
                            reject(new Error(sErrorMsg));
                        }.bind(this)
                    });
                }.bind(this));
            }.bind(this)).then(function(data) {
                oTargetDialog.setBusy(false);
                this._updateComponentList(data, oTargetDialog);
            }.bind(this)).catch(function(error) {
                oTargetDialog.setBusy(false);
                MessageBox.error(error.message);
            });
        },

        /**
         * @function _initializeTestHarness
         * @description Initializes test harness with template test configuration.
         * @param {sap.ui.model.Context} oContext - Selected template context
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _initializeTestHarness: function(oContext, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["testHarness"];
            if (!oTargetDialog) return;
            
            oTargetDialog.setBusy(true);
            const sTemplateId = oContext.getObject().templateId;
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    var oModel = this.base.getView().getModel();
                    return SecurityUtils.secureCallFunction(oModel, "/GetTestConfiguration", {
                        urlParameters: {
                            templateId: sTemplateId
                        },
                        success: function(data) {
                            resolve(data);
                        },
                        error: function(error) {
                            var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                            var sErrorMsg = oBundle.getText("error.loadingTestConfiguration") || "Error loading test configuration";
                            reject(new Error(sErrorMsg));
                        }.bind(this)
                    });
                }.bind(this));
            }.bind(this)).then(function(data) {
                oTargetDialog.setBusy(false);
                this._setupTestHarness(data, oTargetDialog);
            }.bind(this)).catch(function(error) {
                oTargetDialog.setBusy(false);
                MessageBox.error(error.message);
            });
        },

        /**
         * @function _startBatchBuild
         * @description Starts batch build process for multiple templates.
         * @param {Array<sap.ui.model.Context>} aSelectedContexts - Selected template contexts
         * @private
         */
        _startBatchBuild: function(aSelectedContexts) {
            const aTemplateIds = aSelectedContexts.map(ctx => ctx.getObject().templateId);
            
            MessageToast.show(this.getResourceBundle().getText("msg.batchBuildStarted", [aTemplateIds.length]));
            
            if (!SecurityUtils.checkBuilderAuth('StartBatchBuild', {})) {
                return;
            }
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    var oModel = this.base.getView().getModel();
                    return SecurityUtils.secureCallFunction(oModel, "/StartBatchBuild", {
                        urlParameters: {
                            templateIds: aTemplateIds.join(',')
                        },
                        success: function(data) {
                            resolve(data);
                        },
                        error: function(error) {
                            reject(error);
                        }
                    });
                }.bind(this));
            }.bind(this)).then(function(data) {
                MessageToast.show(this.getResourceBundle().getText("msg.batchBuildQueued"));
                this._refreshTemplateData();
            }.bind(this)).catch(function(error) {
                MessageToast.show(this.getResourceBundle().getText("error.batchBuildFailed"));
            });
        },

        /**
         * @function _refreshTemplateData
         * @description Refreshes agent template data in the list.
         * @private
         */
        _refreshTemplateData: function() {
            const oBinding = this.base.getView().byId("fe::table::AgentTemplates::LineItem").getBinding("rows");
            if (oBinding) {
                oBinding.refresh();
            }
        },

        /**
         * @function _updateDashboardCharts
         * @description Updates builder dashboard charts with deployment and build metrics.
         * @param {Object} data - Dashboard data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _updateDashboardCharts: function(data, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["builderDashboard"];
            if (!oTargetDialog) return;
            
            this._createDeploymentTrendsChart(data.deploymentTrends, oTargetDialog);
            this._createBuildMetricsChart(data.buildMetrics, oTargetDialog);
            this._createTemplateUsageChart(data.templateUsage, oTargetDialog);
        },

        /**
         * @function _setupCodeGenerator
         * @description Sets up code generator with template configuration.
         * @param {Object} data - Template details
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _setupCodeGenerator: function(data, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["codeGenerator"];
            if (!oTargetDialog) return;
            
            var oCodeGenModel = new JSONModel({
                templateDetails: data.template,
                codeOptions: data.codeOptions,
                languageConfigs: data.languageConfigs,
                frameworks: data.frameworks,
                selectedLanguage: data.template.codeLanguage,
                selectedFramework: data.template.frameworkVersion
            });
            oTargetDialog.setModel(oCodeGenModel, "codeGen");
        },

        /**
         * @function _updateDeploymentTargets
         * @description Updates deployment target options and configurations.
         * @param {Object} data - Deployment data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _updateDeploymentTargets: function(data, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["deploymentManager"];
            if (!oTargetDialog) return;
            
            var oDeployModel = new JSONModel({
                targets: data.deploymentTargets,
                environments: data.environments,
                configurations: data.configurations,
                activeDeployments: data.activeDeployments
            });
            oTargetDialog.setModel(oDeployModel, "deployment");
        },

        /**
         * @function _updatePipelineList
         * @description Updates build pipeline list and status.
         * @param {Object} data - Pipeline data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _updatePipelineList: function(data, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["pipelineManager"];
            if (!oTargetDialog) return;
            
            var oPipelineModel = new JSONModel({
                pipelines: data.pipelines,
                pipelineTypes: data.pipelineTypes,
                triggers: data.triggers,
                statistics: data.statistics
            });
            oTargetDialog.setModel(oPipelineModel, "pipeline");
        },

        /**
         * @function _updateComponentList
         * @description Updates agent component list for modular design.
         * @param {Object} data - Component data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _updateComponentList: function(data, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["componentBuilder"];
            if (!oTargetDialog) return;
            
            var oComponentModel = new JSONModel({
                components: data.components,
                componentTypes: data.componentTypes,
                dependencies: data.dependencies,
                integrationPoints: data.integrationPoints
            });
            oTargetDialog.setModel(oComponentModel, "component");
        },

        /**
         * @function _setupTestHarness
         * @description Sets up test harness with test configurations.
         * @param {Object} data - Test configuration data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _setupTestHarness: function(data, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["testHarness"];
            if (!oTargetDialog) return;
            
            var oTestModel = new JSONModel({
                testConfiguration: data.configuration,
                testFrameworks: data.frameworks,
                testSuites: data.testSuites,
                coverageTargets: data.coverageTargets,
                testResults: data.recentResults
            });
            oTargetDialog.setModel(oTestModel, "test");
        },

        /**
         * @function _updatePipelineStatus
         * @description Updates pipeline status in real-time.
         * @param {Object} pipeline - Pipeline status data
         * @private
         */
        _updatePipelineStatus: function(pipeline) {
            // Update pipeline status in UI
            var oPipelineDialog = this._dialogCache["pipelineManager"];
            if (oPipelineDialog && oPipelineDialog.isOpen()) {
                var oPipelineModel = oPipelineDialog.getModel("pipeline");
                if (oPipelineModel) {
                    var oData = oPipelineModel.getData();
                    var iPipelineIndex = oData.pipelines.findIndex(p => p.id === pipeline.id);
                    if (iPipelineIndex >= 0) {
                        oData.pipelines[iPipelineIndex] = pipeline;
                        oPipelineModel.setData(oData);
                    }
                }
            }
        },

        /**
         * @function _createDeploymentTrendsChart
         * @description Creates deployment trends chart for dashboard.
         * @param {Object} data - Chart data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createDeploymentTrendsChart: function(data, oDialog) {
            var oChartContainer = oDialog.byId("deploymentTrendsChart");
            if (!oChartContainer || !data) return;
            
            var oChartModel = new JSONModel({
                chartData: data,
                config: {
                    title: this.getResourceBundle().getText("chart.deploymentTrends"),
                    xAxisLabel: this.getResourceBundle().getText("chart.time"),
                    yAxisLabel: this.getResourceBundle().getText("field.deploymentCount")
                }
            });
            oChartContainer.setModel(oChartModel, "chart");
        },
        
        /**
         * @function _createBuildMetricsChart
         * @description Creates build metrics chart for dashboard.
         * @param {Object} data - Chart data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createBuildMetricsChart: function(data, oDialog) {
            var oChartContainer = oDialog.byId("buildMetricsChart");
            if (!oChartContainer || !data) return;
            
            var oChartModel = new JSONModel({
                chartData: data,
                config: {
                    title: this.getResourceBundle().getText("chart.buildMetrics"),
                    showLegend: true,
                    colorPalette: ["#5cbae6", "#b6d7a8", "#ffd93d", "#ff7b7b"]
                }
            });
            oChartContainer.setModel(oChartModel, "chart");
        },
        
        /**
         * @function _createTemplateUsageChart
         * @description Creates template usage chart for analytics.
         * @param {Object} data - Chart data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createTemplateUsageChart: function(data, oDialog) {
            var oChartContainer = oDialog.byId("templateUsageChart");
            if (!oChartContainer || !data) return;
            
            var oChartModel = new JSONModel({
                chartData: data,
                config: {
                    title: this.getResourceBundle().getText("chart.templateUsage"),
                    showDataLabels: true,
                    enableDrillDown: true
                }
            });
            oChartContainer.setModel(oChartModel, "chart");
        },
        
        /**
         * @function _cleanupResources
         * @description Cleans up resources and connections on controller destruction.
         * @private
         */
        _cleanupResources: function() {
            // Close WebSocket connection
            if (this._ws) {
                this._ws.close();
                this._ws = null;
            }
            
            // Clear polling interval
            if (this._pollInterval) {
                clearInterval(this._pollInterval);
                this._pollInterval = null;
            }
            
            // Clear cached dialogs
            for (var sDialogId in this._dialogCache) {
                var oDialog = this._dialogCache[sDialogId];
                if (oDialog && oDialog.destroy) {
                    oDialog.destroy();
                }
            }
            this._dialogCache = {};
            
            // Clear throttled and debounced functions
            if (this._throttledDashboardUpdate) {
                this._throttledDashboardUpdate = null;
            }
            if (this._debouncedSearch) {
                this._debouncedSearch = null;
            }
        },

        /**
         * @function getResourceBundle
         * @description Gets the i18n resource bundle for text translations.
         * @returns {sap.ui.model.resource.ResourceModel} Resource bundle
         * @public
         */
        getResourceBundle: function() {
            return this.base.getView().getModel("i18n").getResourceBundle();
        }
    });
});