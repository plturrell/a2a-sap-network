sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Fragment"
], function(ControllerExtension, MessageToast, MessageBox, Fragment) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent14.ext.controller.ObjectPageExt", {
        
        // Fine-Tune Model Action
        onFineTuneModel: function() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            
            if (oData.status === 'training') {
                MessageToast.show(this.getResourceBundle().getText("msg.alreadyTraining"));
                return;
            }

            if (!this._fineTuningDialog) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent14.ext.fragment.FineTuningDialog",
                    controller: this
                }).then(function(oDialog) {
                    this._fineTuningDialog = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadFineTuningOptions(oContext);
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadFineTuningOptions(oContext);
                this._fineTuningDialog.open();
            }
        },

        // Evaluate Model Action
        onEvaluateModel: function() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            
            if (oData.status === 'training') {
                MessageToast.show(this.getResourceBundle().getText("error.cannotEvaluateWhileTraining"));
                return;
            }

            if (!this._evaluationDialog) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent14.ext.fragment.EvaluationDialog",
                    controller: this
                }).then(function(oDialog) {
                    this._evaluationDialog = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadEvaluationOptions(oContext);
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadEvaluationOptions(oContext);
                this._evaluationDialog.open();
            }
        },

        // Optimize Model Action
        onOptimizeModel: function() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            
            if (oData.status === 'training' || oData.status === 'optimizing') {
                MessageToast.show(this.getResourceBundle().getText("error.cannotOptimizeNow"));
                return;
            }

            if (!this._optimizationDialog) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent14.ext.fragment.OptimizationDialog",
                    controller: this
                }).then(function(oDialog) {
                    this._optimizationDialog = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadOptimizationOptions(oContext);
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadOptimizationOptions(oContext);
                this._optimizationDialog.open();
            }
        },

        // Deploy Model Action
        onDeployModel: function() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            
            if (oData.status !== 'completed' && oData.status !== 'deployed') {
                MessageToast.show(this.getResourceBundle().getText("error.modelNotReady"));
                return;
            }

            MessageBox.confirm(
                this.getResourceBundle().getText("msg.deployModelConfirm"),
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._deployModel(oContext);
                        }
                    }.bind(this)
                }
            );
        },

        // Test Model Action
        onTestModel: function() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            
            if (oData.status === 'training') {
                MessageToast.show(this.getResourceBundle().getText("error.cannotTestWhileTraining"));
                return;
            }

            if (!this._testModelDialog) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent14.ext.fragment.TestModelDialog",
                    controller: this
                }).then(function(oDialog) {
                    this._testModelDialog = oDialog;
                    this.getView().addDependent(oDialog);
                    oDialog.open();
                }.bind(this));
            } else {
                this._testModelDialog.open();
            }
        },

        // Compare Models Action
        onCompareModels: function() {
            const oContext = this.base.getView().getBindingContext();
            
            if (!this._modelComparisonDialog) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent14.ext.fragment.ModelComparison",
                    controller: this
                }).then(function(oDialog) {
                    this._modelComparisonDialog = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadComparisonData(oContext);
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadComparisonData(oContext);
                this._modelComparisonDialog.open();
            }
        },

        // Export Model Action
        onExportModel: function() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            
            if (oData.status === 'training') {
                MessageToast.show(this.getResourceBundle().getText("error.cannotExportWhileTraining"));
                return;
            }

            if (!this._exportDialog) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent14.ext.fragment.ExportModel",
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

        // Visualize Embeddings Action
        onVisualizeEmbeddings: function() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            
            if (!oData.modelPath) {
                MessageToast.show(this.getResourceBundle().getText("error.modelNotTrained"));
                return;
            }

            if (!this._embeddingVisualizer) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent14.ext.fragment.EmbeddingVisualizer",
                    controller: this
                }).then(function(oDialog) {
                    this._embeddingVisualizer = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadEmbeddingData(oContext);
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadEmbeddingData(oContext);
                this._embeddingVisualizer.open();
            }
        },

        // Real-time monitoring initialization
        onAfterRendering: function() {
            this._initializeModelMonitoring();
        },

        _initializeModelMonitoring: function() {
            const oContext = this.base.getView().getBindingContext();
            if (!oContext) return;

            const modelId = oContext.getObject().modelId;
            
            // Subscribe to model updates for this specific model
            if (this._eventSource) {
                this._eventSource.close();
            }

            try {
                this._eventSource = new EventSource(`http://localhost:8014/embedding/${modelId}/stream`);
                
                this._eventSource.addEventListener('training-progress', (event) => {
                    const data = JSON.parse(event.data);
                    this._updateTrainingProgress(data);
                });

                this._eventSource.addEventListener('training-completed', (event) => {
                    const data = JSON.parse(event.data);
                    this._handleTrainingCompleted(data);
                });

                this._eventSource.addEventListener('evaluation-progress', (event) => {
                    const data = JSON.parse(event.data);
                    this._updateEvaluationProgress(data);
                });

                this._eventSource.addEventListener('optimization-progress', (event) => {
                    const data = JSON.parse(event.data);
                    this._updateOptimizationProgress(data);
                });

            } catch (error) {
                console.warn("Server-Sent Events not available, using polling");
                this._initializePolling(modelId);
            }
        },

        _initializePolling: function(modelId) {
            this._pollInterval = setInterval(() => {
                this._refreshModelData();
            }, 3000);
        },

        _loadFineTuningOptions: function(oContext) {
            const oModel = this.getView().getModel();
            const sModelId = oContext.getObject().modelId;
            
            oModel.callFunction("/GetFineTuningOptions", {
                urlParameters: {
                    modelId: sModelId
                },
                success: function(data) {
                    this._displayFineTuningOptions(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingFineTuningOptions"));
                }.bind(this)
            });
        },

        _loadEvaluationOptions: function(oContext) {
            const oModel = this.getView().getModel();
            const sModelId = oContext.getObject().modelId;
            
            oModel.callFunction("/GetEvaluationOptions", {
                urlParameters: {
                    modelId: sModelId
                },
                success: function(data) {
                    this._displayEvaluationOptions(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingEvaluationOptions"));
                }.bind(this)
            });
        },

        _loadOptimizationOptions: function(oContext) {
            const oModel = this.getView().getModel();
            const sModelId = oContext.getObject().modelId;
            
            oModel.callFunction("/GetOptimizationOptions", {
                urlParameters: {
                    modelId: sModelId
                },
                success: function(data) {
                    this._displayOptimizationOptions(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingOptimizationOptions"));
                }.bind(this)
            });
        },

        _deployModel: function(oContext) {
            const oModel = this.getView().getModel();
            const sModelId = oContext.getObject().modelId;
            
            MessageToast.show(this.getResourceBundle().getText("msg.deploymentStarted"));
            
            oModel.callFunction("/DeployModel", {
                urlParameters: {
                    modelId: sModelId
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.modelDeployed"));
                    this._refreshModelData();
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.deploymentFailed"));
                }.bind(this)
            });
        },

        _loadComparisonData: function(oContext) {
            const oModel = this.getView().getModel();
            const sModelId = oContext.getObject().modelId;
            
            oModel.callFunction("/GetModelComparisons", {
                urlParameters: {
                    modelId: sModelId
                },
                success: function(data) {
                    this._displayModelComparisons(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingComparisonData"));
                }.bind(this)
            });
        },

        _loadEmbeddingData: function(oContext) {
            const oModel = this.getView().getModel();
            const sModelId = oContext.getObject().modelId;
            
            oModel.callFunction("/GetEmbeddingVisualization", {
                urlParameters: {
                    modelId: sModelId
                },
                success: function(data) {
                    this._displayEmbeddingVisualization(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingEmbeddingData"));
                }.bind(this)
            });
        },

        _updateTrainingProgress: function(data) {
            // Update training progress indicators
            const oProgressIndicator = this.getView().byId("trainingProgress");
            if (oProgressIndicator) {
                oProgressIndicator.setPercentValue(data.progress);
                oProgressIndicator.setDisplayValue(`Epoch ${data.epoch}/${data.totalEpochs} - Loss: ${data.loss.toFixed(4)}`);
            }
        },

        _handleTrainingCompleted: function(data) {
            MessageToast.show(this.getResourceBundle().getText("msg.fineTuningCompleted"));
            this._refreshModelData();
        },

        _updateEvaluationProgress: function(data) {
            // Update evaluation progress indicators
        },

        _updateOptimizationProgress: function(data) {
            // Update optimization progress indicators
        },

        _refreshModelData: function() {
            const oContext = this.base.getView().getBindingContext();
            if (oContext) {
                oContext.refresh();
            }
        },

        _displayFineTuningOptions: function(data) {
            // Display fine-tuning options in dialog
        },

        _displayEvaluationOptions: function(data) {
            // Display evaluation options in dialog
        },

        _displayOptimizationOptions: function(data) {
            // Display optimization options in dialog
        },

        _displayModelComparisons: function(data) {
            // Display model comparison results
        },

        _displayEmbeddingVisualization: function(data) {
            // Display embedding visualization
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