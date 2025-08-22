sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Fragment"
], function(ControllerExtension, MessageToast, MessageBox, Fragment) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent10.ext.controller.ListReportExt", {
        
        // Calculation Dashboard Action
        onCalculationDashboard: function() {
            if (!this._calculationDashboard) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent10.ext.fragment.CalculationDashboard",
                    controller: this
                }).then(function(oDialog) {
                    this._calculationDashboard = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadDashboardData();
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadDashboardData();
                this._calculationDashboard.open();
            }
        },

        // Create New Calculation Task
        onCreateCalculationTask: function() {
            if (!this._createTaskDialog) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent10.ext.fragment.CreateCalculationTask",
                    controller: this
                }).then(function(oDialog) {
                    this._createTaskDialog = oDialog;
                    this.getView().addDependent(oDialog);
                    oDialog.open();
                }.bind(this));
            } else {
                this._createTaskDialog.open();
            }
        },

        // Formula Builder Action
        onFormulaBuilder: function() {
            if (!this._formulaBuilder) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent10.ext.fragment.FormulaBuilder",
                    controller: this
                }).then(function(oDialog) {
                    this._formulaBuilder = oDialog;
                    this.getView().addDependent(oDialog);
                    oDialog.open();
                }.bind(this));
            } else {
                this._formulaBuilder.open();
            }
        },

        // Statistical Analyzer Action
        onStatisticalAnalyzer: function() {
            if (!this._statisticalAnalyzer) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent10.ext.fragment.StatisticalAnalyzer",
                    controller: this
                }).then(function(oDialog) {
                    this._statisticalAnalyzer = oDialog;
                    this.getView().addDependent(oDialog);
                    oDialog.open();
                }.bind(this));
            } else {
                this._statisticalAnalyzer.open();
            }
        },

        // Engine Manager Action
        onEngineManager: function() {
            if (!this._engineManager) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent10.ext.fragment.EngineManager",
                    controller: this
                }).then(function(oDialog) {
                    this._engineManager = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadEngineData();
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadEngineData();
                this._engineManager.open();
            }
        },

        // Precision Validator Action
        onPrecisionValidator: function() {
            const oBinding = this.base.getView().byId("fe::table::CalculationTasks::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectTasksFirst"));
                return;
            }

            this._runPrecisionValidation(aSelectedContexts);
        },

        // Batch Calculator Action
        onBatchCalculator: function() {
            const oBinding = this.base.getView().byId("fe::table::CalculationTasks::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectTasksFirst"));
                return;
            }

            MessageBox.confirm(
                this.getResourceBundle().getText("msg.batchCalculationConfirm", [aSelectedContexts.length]),
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._runBatchCalculation(aSelectedContexts);
                        }
                    }.bind(this)
                }
            );
        },

        // Performance Optimizer Action
        onPerformanceOptimizer: function() {
            if (!this._performanceOptimizer) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent10.ext.fragment.PerformanceOptimizer",
                    controller: this
                }).then(function(oDialog) {
                    this._performanceOptimizer = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadPerformanceData();
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadPerformanceData();
                this._performanceOptimizer.open();
            }
        },

        // Real-time Updates via WebSocket
        onAfterRendering: function() {
            this._initializeWebSocket();
        },

        _initializeWebSocket: function() {
            if (this._ws) return;

            try {
                this._ws = new WebSocket('ws://localhost:8010/calculations/updates');
                
                this._ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    this._handleCalculationUpdate(data);
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
                this._refreshCalculationData();
            }, 5000);
        },

        _handleCalculationUpdate: function(data) {
            const oModel = this.getView().getModel();
            
            switch (data.type) {
                case 'CALCULATION_STARTED':
                    MessageToast.show(this.getResourceBundle().getText("msg.calculationStarted"));
                    break;
                case 'CALCULATION_COMPLETED':
                    MessageToast.show(this.getResourceBundle().getText("msg.calculationCompleted"));
                    this._refreshCalculationData();
                    break;
                case 'CALCULATION_FAILED':
                    MessageToast.show(this.getResourceBundle().getText("error.calculationFailed", [data.error]));
                    break;
                case 'PERFORMANCE_UPDATE':
                    this._updatePerformanceMetrics(data.metrics);
                    break;
            }
        },

        _loadDashboardData: function() {
            const oModel = this.getView().getModel();
            
            // Load calculation statistics
            oModel.callFunction("/GetCalculationStatistics", {
                success: function(data) {
                    this._updateDashboardCharts(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingStatistics"));
                }.bind(this)
            });
        },

        _loadEngineData: function() {
            const oModel = this.getView().getModel();
            
            oModel.callFunction("/GetEngineStatus", {
                success: function(data) {
                    this._updateEngineStatus(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingEngineStatus"));
                }.bind(this)
            });
        },

        _loadPerformanceData: function() {
            const oModel = this.getView().getModel();
            
            oModel.callFunction("/GetPerformanceMetrics", {
                success: function(data) {
                    this._updatePerformanceCharts(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingPerformanceData"));
                }.bind(this)
            });
        },

        _runPrecisionValidation: function(aSelectedContexts) {
            const oModel = this.getView().getModel();
            const aTaskIds = aSelectedContexts.map(ctx => ctx.getObject().taskId);
            
            oModel.callFunction("/ValidatePrecision", {
                urlParameters: {
                    taskIds: aTaskIds.join(',')
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.precisionVerified"));
                    this._refreshCalculationData();
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.precisionValidationFailed"));
                }.bind(this)
            });
        },

        _runBatchCalculation: function(aSelectedContexts) {
            const oModel = this.getView().getModel();
            const aTaskIds = aSelectedContexts.map(ctx => ctx.getObject().taskId);
            
            MessageToast.show(this.getResourceBundle().getText("msg.batchCalculationStarted", [aTaskIds.length]));
            
            oModel.callFunction("/ExecuteBatchCalculation", {
                urlParameters: {
                    taskIds: aTaskIds.join(',')
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.batchCalculationCompleted"));
                    this._refreshCalculationData();
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.batchCalculationFailed"));
                }.bind(this)
            });
        },

        _refreshCalculationData: function() {
            const oBinding = this.base.getView().byId("fe::table::CalculationTasks::LineItem").getBinding("rows");
            oBinding.refresh();
        },

        _updateDashboardCharts: function(data) {
            // Update performance trends chart
            // Update accuracy metrics chart
            // Update engine comparison chart
        },

        _updateEngineStatus: function(data) {
            // Update engine status indicators
            // Update resource utilization
        },

        _updatePerformanceCharts: function(data) {
            // Update throughput analysis
            // Update resource efficiency
            // Update scalability metrics
        },

        _updatePerformanceMetrics: function(metrics) {
            // Real-time performance metric updates
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