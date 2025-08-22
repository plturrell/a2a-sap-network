sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Fragment"
], function(ControllerExtension, MessageToast, MessageBox, Fragment) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent13.ext.controller.ListReportExt", {
        
        // Builder Dashboard Action
        onBuilderDashboard: function() {
            if (!this._builderDashboard) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent13.ext.fragment.BuilderDashboard",
                    controller: this
                }).then(function(oDialog) {
                    this._builderDashboard = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadDashboardData();
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadDashboardData();
                this._builderDashboard.open();
            }
        },

        // Create New Agent Template
        onCreateAgentTemplate: function() {
            if (!this._templateWizard) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent13.ext.fragment.TemplateWizard",
                    controller: this
                }).then(function(oDialog) {
                    this._templateWizard = oDialog;
                    this.getView().addDependent(oDialog);
                    oDialog.open();
                }.bind(this));
            } else {
                this._templateWizard.open();
            }
        },

        // Code Generator Action
        onCodeGenerator: function() {
            const oBinding = this.base.getView().byId("fe::table::AgentTemplates::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectTemplatesFirst"));
                return;
            }

            if (!this._codeGenerator) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent13.ext.fragment.CodeGenerator",
                    controller: this
                }).then(function(oDialog) {
                    this._codeGenerator = oDialog;
                    this.getView().addDependent(oDialog);
                    this._initializeCodeGenerator(aSelectedContexts[0]);
                    oDialog.open();
                }.bind(this));
            } else {
                this._initializeCodeGenerator(aSelectedContexts[0]);
                this._codeGenerator.open();
            }
        },

        // Deployment Manager Action
        onDeploymentManager: function() {
            if (!this._deploymentManager) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent13.ext.fragment.DeploymentManager",
                    controller: this
                }).then(function(oDialog) {
                    this._deploymentManager = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadDeploymentData();
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadDeploymentData();
                this._deploymentManager.open();
            }
        },

        // Pipeline Manager Action
        onPipelineManager: function() {
            if (!this._pipelineManager) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent13.ext.fragment.PipelineManager",
                    controller: this
                }).then(function(oDialog) {
                    this._pipelineManager = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadPipelineData();
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadPipelineData();
                this._pipelineManager.open();
            }
        },

        // Component Builder Action
        onComponentBuilder: function() {
            const oBinding = this.base.getView().byId("fe::table::AgentTemplates::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectTemplatesFirst"));
                return;
            }

            if (!this._componentBuilder) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent13.ext.fragment.ComponentBuilder",
                    controller: this
                }).then(function(oDialog) {
                    this._componentBuilder = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadComponentData(aSelectedContexts[0]);
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadComponentData(aSelectedContexts[0]);
                this._componentBuilder.open();
            }
        },

        // Test Harness Action
        onTestHarness: function() {
            const oBinding = this.base.getView().byId("fe::table::AgentTemplates::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectTemplatesFirst"));
                return;
            }

            if (!this._testHarness) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent13.ext.fragment.TestHarness",
                    controller: this
                }).then(function(oDialog) {
                    this._testHarness = oDialog;
                    this.getView().addDependent(oDialog);
                    this._initializeTestHarness(aSelectedContexts[0]);
                    oDialog.open();
                }.bind(this));
            } else {
                this._initializeTestHarness(aSelectedContexts[0]);
                this._testHarness.open();
            }
        },

        // Batch Build Action
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

        // Real-time Updates via WebSocket
        onAfterRendering: function() {
            this._initializeWebSocket();
        },

        _initializeWebSocket: function() {
            if (this._ws) return;

            try {
                this._ws = new WebSocket('ws://localhost:8013/builder/updates');
                
                this._ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    this._handleBuilderUpdate(data);
                }.bind(this);

                this._ws.onclose = function() {
                    setTimeout(() => this._initializeWebSocket(), 5000);
                }.bind(this);

            } catch (error) {
                console.warn("WebSocket connection failed, falling back to polling");
                this._initializePolling();
            }
        },

        _initializePolling: function() {
            this._pollInterval = setInterval(() => {
                this._refreshTemplateData();
            }, 5000);
        },

        _handleBuilderUpdate: function(data) {
            const oModel = this.getView().getModel();
            
            switch (data.type) {
                case 'BUILD_STARTED':
                    MessageToast.show(this.getResourceBundle().getText("msg.buildStarted"));
                    break;
                case 'BUILD_COMPLETED':
                    MessageToast.show(this.getResourceBundle().getText("msg.buildCompleted"));
                    this._refreshTemplateData();
                    break;
                case 'BUILD_FAILED':
                    MessageToast.show(this.getResourceBundle().getText("error.buildFailed"));
                    break;
                case 'DEPLOYMENT_STARTED':
                    MessageToast.show(this.getResourceBundle().getText("msg.deploymentStarted"));
                    break;
                case 'DEPLOYMENT_COMPLETED':
                    MessageToast.show(this.getResourceBundle().getText("msg.deploymentCompleted"));
                    this._refreshTemplateData();
                    break;
                case 'DEPLOYMENT_FAILED':
                    MessageToast.show(this.getResourceBundle().getText("error.deploymentFailed"));
                    break;
                case 'PIPELINE_UPDATE':
                    this._updatePipelineStatus(data.pipeline);
                    break;
            }
        },

        _loadDashboardData: function() {
            const oModel = this.getView().getModel();
            
            // Load builder statistics
            oModel.callFunction("/GetBuilderStatistics", {
                success: function(data) {
                    this._updateDashboardCharts(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingStatistics"));
                }.bind(this)
            });
        },

        _initializeCodeGenerator: function(oContext) {
            const oModel = this.getView().getModel();
            const sTemplateId = oContext.getObject().templateId;
            
            oModel.callFunction("/GetTemplateDetails", {
                urlParameters: {
                    templateId: sTemplateId
                },
                success: function(data) {
                    this._setupCodeGenerator(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingTemplateDetails"));
                }.bind(this)
            });
        },

        _loadDeploymentData: function() {
            const oModel = this.getView().getModel();
            
            oModel.callFunction("/GetDeploymentTargets", {
                success: function(data) {
                    this._updateDeploymentTargets(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingDeploymentData"));
                }.bind(this)
            });
        },

        _loadPipelineData: function() {
            const oModel = this.getView().getModel();
            
            oModel.callFunction("/GetBuildPipelines", {
                success: function(data) {
                    this._updatePipelineList(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingPipelineData"));
                }.bind(this)
            });
        },

        _loadComponentData: function(oContext) {
            const oModel = this.getView().getModel();
            const sTemplateId = oContext.getObject().templateId;
            
            oModel.callFunction("/GetAgentComponents", {
                urlParameters: {
                    templateId: sTemplateId
                },
                success: function(data) {
                    this._updateComponentList(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingComponentData"));
                }.bind(this)
            });
        },

        _initializeTestHarness: function(oContext) {
            const oModel = this.getView().getModel();
            const sTemplateId = oContext.getObject().templateId;
            
            oModel.callFunction("/GetTestConfiguration", {
                urlParameters: {
                    templateId: sTemplateId
                },
                success: function(data) {
                    this._setupTestHarness(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingTestConfiguration"));
                }.bind(this)
            });
        },

        _startBatchBuild: function(aSelectedContexts) {
            const oModel = this.getView().getModel();
            const aTemplateIds = aSelectedContexts.map(ctx => ctx.getObject().templateId);
            
            MessageToast.show(this.getResourceBundle().getText("msg.batchBuildStarted", [aTemplateIds.length]));
            
            oModel.callFunction("/StartBatchBuild", {
                urlParameters: {
                    templateIds: aTemplateIds.join(',')
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.batchBuildQueued"));
                    this._refreshTemplateData();
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.batchBuildFailed"));
                }.bind(this)
            });
        },

        _refreshTemplateData: function() {
            const oBinding = this.base.getView().byId("fe::table::AgentTemplates::LineItem").getBinding("rows");
            oBinding.refresh();
        },

        _updateDashboardCharts: function(data) {
            // Update deployment trends chart
            // Update build metrics chart
            // Update template usage chart
        },

        _setupCodeGenerator: function(data) {
            // Setup code generator with template details
        },

        _updateDeploymentTargets: function(data) {
            // Update deployment target options
        },

        _updatePipelineList: function(data) {
            // Update pipeline list
        },

        _updateComponentList: function(data) {
            // Update component list
        },

        _setupTestHarness: function(data) {
            // Setup test harness configuration
        },

        _updatePipelineStatus: function(pipeline) {
            // Update pipeline status in real-time
        },

        getResourceBundle: function() {
            return this.getView().getModel("i18n").getResourceBundle();
        },

        onExit: function() {
            if (this._ws) {
                this._ws.close();
            }
            if (this._pollInterval) {
                clearInterval(this._pollInterval);
            }
        }
    });
});