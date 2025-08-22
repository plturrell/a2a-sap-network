sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Fragment"
], function(ControllerExtension, MessageToast, MessageBox, Fragment) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent13.ext.controller.ObjectPageExt", {
        
        // Generate Agent Action
        onGenerateAgent: function() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            
            if (oData.deploymentStatus === 'deploying' || oData.deploymentStatus === 'deployed') {
                MessageToast.show(this.getResourceBundle().getText("msg.agentAlreadyDeployed"));
                return;
            }

            MessageBox.confirm(
                this.getResourceBundle().getText("msg.generateAgentConfirm"),
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._generateAgent(oContext);
                        }
                    }.bind(this)
                }
            );
        },

        // Deploy Agent Action
        onDeployAgent: function() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            
            if (oData.deploymentStatus !== 'ready') {
                MessageToast.show(this.getResourceBundle().getText("error.notReadyForDeployment"));
                return;
            }

            if (!this._deploymentDialog) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent13.ext.fragment.DeploymentDialog",
                    controller: this
                }).then(function(oDialog) {
                    this._deploymentDialog = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadDeploymentOptions(oContext);
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadDeploymentOptions(oContext);
                this._deploymentDialog.open();
            }
        },

        // Test Agent Action
        onTestAgent: function() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            
            if (!oData.codeGenerated) {
                MessageToast.show(this.getResourceBundle().getText("error.codeNotGenerated"));
                return;
            }

            if (!this._testRunner) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent13.ext.fragment.TestRunner",
                    controller: this
                }).then(function(oDialog) {
                    this._testRunner = oDialog;
                    this.getView().addDependent(oDialog);
                    this._initializeTestRunner(oContext);
                    oDialog.open();
                }.bind(this));
            } else {
                this._initializeTestRunner(oContext);
                this._testRunner.open();
            }
        },

        // Build Agent Action
        onBuildAgent: function() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            
            if (!oData.codeGenerated) {
                MessageToast.show(this.getResourceBundle().getText("error.codeNotGenerated"));
                return;
            }

            MessageBox.confirm(
                this.getResourceBundle().getText("msg.buildAgentConfirm"),
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._buildAgent(oContext);
                        }
                    }.bind(this)
                }
            );
        },

        // Configure Agent Action
        onConfigureAgent: function() {
            const oContext = this.base.getView().getBindingContext();
            
            if (!this._configurationEditor) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent13.ext.fragment.ConfigurationEditor",
                    controller: this
                }).then(function(oDialog) {
                    this._configurationEditor = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadConfiguration(oContext);
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadConfiguration(oContext);
                this._configurationEditor.open();
            }
        },

        // Clone Template Action
        onCloneTemplate: function() {
            const oContext = this.base.getView().getBindingContext();
            
            MessageBox.confirm(
                this.getResourceBundle().getText("msg.cloneTemplateConfirm"),
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._cloneTemplate(oContext);
                        }
                    }.bind(this)
                }
            );
        },

        // Export Template Action
        onExportTemplate: function() {
            const oContext = this.base.getView().getBindingContext();
            
            if (!this._exportDialog) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent13.ext.fragment.ExportTemplate",
                    controller: this
                }).then(function(oDialog) {
                    this._exportDialog = oDialog;
                    this.getView().addDependent(oDialog);
                    oDialog.open();
                }.bind(this));
            } else {
                this._exportDialog.open();
            }
        },

        // Validate Template Action
        onValidateTemplate: function() {
            const oContext = this.base.getView().getBindingContext();
            
            this._validateTemplate(oContext);
        },

        // Real-time monitoring initialization
        onAfterRendering: function() {
            this._initializeAgentMonitoring();
        },

        _initializeAgentMonitoring: function() {
            const oContext = this.base.getView().getBindingContext();
            if (!oContext) return;

            const templateId = oContext.getObject().templateId;
            
            // Subscribe to agent builder updates for this specific template
            if (this._eventSource) {
                this._eventSource.close();
            }

            try {
                this._eventSource = new EventSource(`http://localhost:8013/builder/${templateId}/stream`);
                
                this._eventSource.addEventListener('generation-progress', (event) => {
                    const data = JSON.parse(event.data);
                    this._updateGenerationProgress(data);
                });

                this._eventSource.addEventListener('build-progress', (event) => {
                    const data = JSON.parse(event.data);
                    this._updateBuildProgress(data);
                });

                this._eventSource.addEventListener('deployment-progress', (event) => {
                    const data = JSON.parse(event.data);
                    this._updateDeploymentProgress(data);
                });

                this._eventSource.addEventListener('test-progress', (event) => {
                    const data = JSON.parse(event.data);
                    this._updateTestProgress(data);
                });

            } catch (error) {
                console.warn("Server-Sent Events not available, using polling");
                this._initializePolling(templateId);
            }
        },

        _initializePolling: function(templateId) {
            this._pollInterval = setInterval(() => {
                this._refreshTemplateData();
            }, 3000);
        },

        _generateAgent: function(oContext) {
            const oModel = this.getView().getModel();
            const sTemplateId = oContext.getObject().templateId;
            
            MessageToast.show(this.getResourceBundle().getText("msg.generationStarted"));
            
            oModel.callFunction("/GenerateAgent", {
                urlParameters: {
                    templateId: sTemplateId
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.agentGenerated"));
                    this._refreshTemplateData();
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.generationFailed"));
                }.bind(this)
            });
        },

        _buildAgent: function(oContext) {
            const oModel = this.getView().getModel();
            const sTemplateId = oContext.getObject().templateId;
            
            MessageToast.show(this.getResourceBundle().getText("msg.buildStarted"));
            
            oModel.callFunction("/BuildAgent", {
                urlParameters: {
                    templateId: sTemplateId
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.buildCompleted"));
                    this._refreshTemplateData();
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.buildFailed"));
                }.bind(this)
            });
        },

        _validateTemplate: function(oContext) {
            const oModel = this.getView().getModel();
            const sTemplateId = oContext.getObject().templateId;
            
            MessageToast.show(this.getResourceBundle().getText("msg.validationStarted"));
            
            oModel.callFunction("/ValidateTemplate", {
                urlParameters: {
                    templateId: sTemplateId
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.templateValidated"));
                    this._refreshTemplateData();
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.validationFailed"));
                }.bind(this)
            });
        },

        _cloneTemplate: function(oContext) {
            const oModel = this.getView().getModel();
            const sTemplateId = oContext.getObject().templateId;
            
            oModel.callFunction("/CloneTemplate", {
                urlParameters: {
                    templateId: sTemplateId
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.templateCloned"));
                    this._refreshTemplateData();
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.cloneFailed"));
                }.bind(this)
            });
        },

        _loadDeploymentOptions: function(oContext) {
            const oModel = this.getView().getModel();
            const sTemplateId = oContext.getObject().templateId;
            
            oModel.callFunction("/GetDeploymentOptions", {
                urlParameters: {
                    templateId: sTemplateId
                },
                success: function(data) {
                    this._displayDeploymentOptions(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingDeploymentOptions"));
                }.bind(this)
            });
        },

        _initializeTestRunner: function(oContext) {
            const oModel = this.getView().getModel();
            const sTemplateId = oContext.getObject().templateId;
            
            oModel.callFunction("/GetTestSuite", {
                urlParameters: {
                    templateId: sTemplateId
                },
                success: function(data) {
                    this._setupTestRunner(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingTestSuite"));
                }.bind(this)
            });
        },

        _loadConfiguration: function(oContext) {
            const oModel = this.getView().getModel();
            const sTemplateId = oContext.getObject().templateId;
            
            oModel.callFunction("/GetAgentConfiguration", {
                urlParameters: {
                    templateId: sTemplateId
                },
                success: function(data) {
                    this._displayConfiguration(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingConfiguration"));
                }.bind(this)
            });
        },

        _updateGenerationProgress: function(data) {
            // Update code generation progress
            const oProgressIndicator = this.getView().byId("generationProgress");
            if (oProgressIndicator) {
                oProgressIndicator.setPercentValue(data.progress);
                oProgressIndicator.setDisplayValue(`${data.progress}% - ${data.currentStep}`);
            }
        },

        _updateBuildProgress: function(data) {
            // Update build progress
            const oProgressIndicator = this.getView().byId("buildProgress");
            if (oProgressIndicator) {
                oProgressIndicator.setPercentValue(data.progress);
                oProgressIndicator.setDisplayValue(`${data.progress}% - ${data.currentStep}`);
            }
        },

        _updateDeploymentProgress: function(data) {
            // Update deployment progress
            const oProgressIndicator = this.getView().byId("deploymentProgress");
            if (oProgressIndicator) {
                oProgressIndicator.setPercentValue(data.progress);
                oProgressIndicator.setDisplayValue(`${data.progress}% - ${data.currentStep}`);
            }
        },

        _updateTestProgress: function(data) {
            // Update test execution progress
            const oProgressIndicator = this.getView().byId("testProgress");
            if (oProgressIndicator) {
                oProgressIndicator.setPercentValue(data.progress);
                oProgressIndicator.setDisplayValue(`${data.progress}% - ${data.testsCompleted}/${data.totalTests}`);
            }
        },

        _refreshTemplateData: function() {
            const oContext = this.base.getView().getBindingContext();
            if (oContext) {
                oContext.refresh();
            }
        },

        _displayDeploymentOptions: function(data) {
            // Display deployment options in dialog
        },

        _setupTestRunner: function(data) {
            // Setup test runner with test suite
        },

        _displayConfiguration: function(data) {
            // Display configuration in editor
        },

        getResourceBundle: function() {
            return this.getView().getModel("i18n").getResourceBundle();
        },

        onExit: function() {
            if (this._eventSource) {
                this._eventSource.close();
            }
            if (this._pollInterval) {
                clearInterval(this._pollInterval);
            }
        }
    });
});