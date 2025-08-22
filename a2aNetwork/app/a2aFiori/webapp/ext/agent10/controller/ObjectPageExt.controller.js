sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Fragment"
], function(ControllerExtension, MessageToast, MessageBox, Fragment) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent10.ext.controller.ObjectPageExt", {
        
        // Execute Calculation Action
        onExecuteCalculation: function() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            
            if (oData.status === 'calculating') {
                MessageToast.show(this.getResourceBundle().getText("msg.calculationAlreadyRunning"));
                return;
            }

            MessageBox.confirm(
                this.getResourceBundle().getText("msg.executeCalculationConfirm"),
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._executeCalculation(oContext);
                        }
                    }.bind(this)
                }
            );
        },

        // Validate Result Action
        onValidateResult: function() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            
            if (!oData.resultValue) {
                MessageToast.show(this.getResourceBundle().getText("error.noResultToValidate"));
                return;
            }

            this._validateCalculationResult(oContext);
        },

        // Optimize Formula Action
        onOptimizeFormula: function() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            
            if (!oData.formulaExpression) {
                MessageToast.show(this.getResourceBundle().getText("error.noFormulaToOptimize"));
                return;
            }

            if (!this._formulaOptimizer) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent10.ext.fragment.FormulaOptimizer",
                    controller: this
                }).then(function(oDialog) {
                    this._formulaOptimizer = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadFormulaOptimizationData(oContext);
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadFormulaOptimizationData(oContext);
                this._formulaOptimizer.open();
            }
        },

        // Analyze Performance Action
        onAnalyzePerformance: function() {
            const oContext = this.base.getView().getBindingContext();
            
            if (!this._performanceAnalyzer) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent10.ext.fragment.PerformanceAnalyzer",
                    controller: this
                }).then(function(oDialog) {
                    this._performanceAnalyzer = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadPerformanceAnalysis(oContext);
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadPerformanceAnalysis(oContext);
                this._performanceAnalyzer.open();
            }
        },

        // Test Precision Action
        onTestPrecision: function() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            
            if (!oData.formulaExpression) {
                MessageToast.show(this.getResourceBundle().getText("error.noFormulaToTest"));
                return;
            }

            this._runPrecisionTest(oContext);
        },

        // Run Self-Healing Action
        onRunSelfHealing: function() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            
            if (!oData.selfHealingEnabled) {
                MessageToast.show(this.getResourceBundle().getText("error.selfHealingDisabled"));
                return;
            }

            MessageBox.confirm(
                this.getResourceBundle().getText("msg.runSelfHealingConfirm"),
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._runSelfHealing(oContext);
                        }
                    }.bind(this)
                }
            );
        },

        // Export Results Action
        onExportResults: function() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            
            if (!oData.resultValue) {
                MessageToast.show(this.getResourceBundle().getText("error.noResultsToExport"));
                return;
            }

            if (!this._resultExporter) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent10.ext.fragment.ResultExporter",
                    controller: this
                }).then(function(oDialog) {
                    this._resultExporter = oDialog;
                    this.getView().addDependent(oDialog);
                    oDialog.open();
                }.bind(this));
            } else {
                this._resultExporter.open();
            }
        },

        // Visualize Data Action
        onVisualizeData: function() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            
            if (!oData.resultValue) {
                MessageToast.show(this.getResourceBundle().getText("error.noDataToVisualize"));
                return;
            }

            if (!this._dataVisualizer) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent10.ext.fragment.DataVisualizer",
                    controller: this
                }).then(function(oDialog) {
                    this._dataVisualizer = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadVisualizationData(oContext);
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadVisualizationData(oContext);
                this._dataVisualizer.open();
            }
        },

        // Real-time monitoring initialization
        onAfterRendering: function() {
            this._initializeCalculationMonitoring();
        },

        _initializeCalculationMonitoring: function() {
            const oContext = this.base.getView().getBindingContext();
            if (!oContext) return;

            const taskId = oContext.getObject().taskId;
            
            // Subscribe to calculation updates for this specific task
            if (this._eventSource) {
                this._eventSource.close();
            }

            try {
                this._eventSource = new EventSource(`http://localhost:8010/calculations/${taskId}/stream`);
                
                this._eventSource.addEventListener('calculation-progress', (event) => {
                    const data = JSON.parse(event.data);
                    this._updateCalculationProgress(data);
                });

                this._eventSource.addEventListener('calculation-completed', (event) => {
                    const data = JSON.parse(event.data);
                    this._handleCalculationCompleted(data);
                });

                this._eventSource.addEventListener('calculation-error', (event) => {
                    const data = JSON.parse(event.data);
                    this._handleCalculationError(data);
                });

                this._eventSource.addEventListener('self-healing', (event) => {
                    const data = JSON.parse(event.data);
                    this._handleSelfHealingUpdate(data);
                });

            } catch (error) {
                console.warn("Server-Sent Events not available, using polling");
                this._initializePolling(taskId);
            }
        },

        _initializePolling: function(taskId) {
            this._pollInterval = setInterval(() => {
                this._refreshTaskData();
            }, 2000);
        },

        _executeCalculation: function(oContext) {
            const oModel = this.getView().getModel();
            const sTaskId = oContext.getObject().taskId;
            
            MessageToast.show(this.getResourceBundle().getText("msg.calculationStarted"));
            
            oModel.callFunction("/ExecuteCalculation", {
                urlParameters: {
                    taskId: sTaskId
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.calculationExecuted"));
                    this._refreshTaskData();
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.calculationFailed", [error.message]));
                }.bind(this)
            });
        },

        _validateCalculationResult: function(oContext) {
            const oModel = this.getView().getModel();
            const sTaskId = oContext.getObject().taskId;
            
            oModel.callFunction("/ValidateResult", {
                urlParameters: {
                    taskId: sTaskId
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.resultValidated"));
                    this._refreshTaskData();
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.resultValidationFailed"));
                }.bind(this)
            });
        },

        _runPrecisionTest: function(oContext) {
            const oModel = this.getView().getModel();
            const sTaskId = oContext.getObject().taskId;
            
            MessageToast.show(this.getResourceBundle().getText("msg.precisionTestStarted"));
            
            oModel.callFunction("/TestPrecision", {
                urlParameters: {
                    taskId: sTaskId
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.precisionTestCompleted"));
                    this._refreshTaskData();
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.precisionTestFailed"));
                }.bind(this)
            });
        },

        _runSelfHealing: function(oContext) {
            const oModel = this.getView().getModel();
            const sTaskId = oContext.getObject().taskId;
            
            MessageToast.show(this.getResourceBundle().getText("msg.selfHealingTriggered"));
            
            oModel.callFunction("/RunSelfHealing", {
                urlParameters: {
                    taskId: sTaskId
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.selfHealingCompleted"));
                    this._refreshTaskData();
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.selfHealingFailed"));
                }.bind(this)
            });
        },

        _loadFormulaOptimizationData: function(oContext) {
            const oModel = this.getView().getModel();
            const sTaskId = oContext.getObject().taskId;
            
            oModel.callFunction("/GetOptimizationSuggestions", {
                urlParameters: {
                    taskId: sTaskId
                },
                success: function(data) {
                    this._displayOptimizationSuggestions(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingOptimizationData"));
                }.bind(this)
            });
        },

        _loadPerformanceAnalysis: function(oContext) {
            const oModel = this.getView().getModel();
            const sTaskId = oContext.getObject().taskId;
            
            oModel.callFunction("/GetPerformanceAnalysis", {
                urlParameters: {
                    taskId: sTaskId
                },
                success: function(data) {
                    this._displayPerformanceAnalysis(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingPerformanceAnalysis"));
                }.bind(this)
            });
        },

        _loadVisualizationData: function(oContext) {
            const oModel = this.getView().getModel();
            const sTaskId = oContext.getObject().taskId;
            
            oModel.callFunction("/GetVisualizationData", {
                urlParameters: {
                    taskId: sTaskId
                },
                success: function(data) {
                    this._createDataVisualization(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingVisualizationData"));
                }.bind(this)
            });
        },

        _updateCalculationProgress: function(data) {
            // Update progress indicators
            const oProgressIndicator = this.getView().byId("calculationProgress");
            if (oProgressIndicator) {
                oProgressIndicator.setPercentValue(data.progress);
                oProgressIndicator.setDisplayValue(`${data.progress}% - ${data.currentStep}`);
            }
        },

        _handleCalculationCompleted: function(data) {
            MessageToast.show(this.getResourceBundle().getText("msg.calculationCompleted"));
            this._refreshTaskData();
            
            // Show performance improvement if available
            if (data.performanceImprovement) {
                MessageToast.show(this.getResourceBundle().getText("msg.performanceImproved", [data.performanceImprovement]));
            }
        },

        _handleCalculationError: function(data) {
            MessageBox.error(this.getResourceBundle().getText("error.calculationFailed", [data.error]));
            this._refreshTaskData();
        },

        _handleSelfHealingUpdate: function(data) {
            MessageToast.show(this.getResourceBundle().getText("msg.selfHealingUpdate", [data.action]));
            this._refreshTaskData();
        },

        _refreshTaskData: function() {
            const oContext = this.base.getView().getBindingContext();
            if (oContext) {
                oContext.refresh();
            }
        },

        _displayOptimizationSuggestions: function(data) {
            // Display optimization suggestions in dialog
        },

        _displayPerformanceAnalysis: function(data) {
            // Display performance analysis charts
        },

        _createDataVisualization: function(data) {
            // Create data visualization charts
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