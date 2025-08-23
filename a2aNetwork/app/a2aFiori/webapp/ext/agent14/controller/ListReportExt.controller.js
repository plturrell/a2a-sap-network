sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Fragment",
    "a2a/network/agent14/ext/utils/SecurityUtils"
], function(ControllerExtension, MessageToast, MessageBox, Fragment, SecurityUtils) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent14.ext.controller.ListReportExt", {
        
        // Fine-Tuning Dashboard Action
        onFineTuningDashboard: function() {
            if (!this._fineTuningDashboard) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent14.ext.fragment.FineTuningDashboard",
                    controller: this
                }).then(function(oDialog) {
                    this._fineTuningDashboard = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadDashboardData();
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadDashboardData();
                this._fineTuningDashboard.open();
            }
        },

        // Create New Embedding Model
        onCreateEmbeddingModel: function() {
            if (!this._createModelDialog) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent14.ext.fragment.CreateEmbeddingModel",
                    controller: this
                }).then(function(oDialog) {
                    this._createModelDialog = oDialog;
                    this.getView().addDependent(oDialog);
                    oDialog.open();
                }.bind(this));
            } else {
                this._createModelDialog.open();
            }
        },

        // Start Fine-Tuning Action
        onStartFineTuning: function() {
            const oBinding = this.base.getView().byId("fe::table::EmbeddingModels::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectModelsFirst"));
                return;
            }

            if (!this._fineTuningWizard) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent14.ext.fragment.FineTuningWizard",
                    controller: this
                }).then(function(oDialog) {
                    this._fineTuningWizard = oDialog;
                    this.getView().addDependent(oDialog);
                    this._initializeFineTuningWizard(aSelectedContexts[0]);
                    oDialog.open();
                }.bind(this));
            } else {
                this._initializeFineTuningWizard(aSelectedContexts[0]);
                this._fineTuningWizard.open();
            }
        },

        // Model Evaluator Action
        onModelEvaluator: function() {
            const oBinding = this.base.getView().byId("fe::table::EmbeddingModels::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectModelsFirst"));
                return;
            }

            if (!this._modelEvaluator) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent14.ext.fragment.ModelEvaluator",
                    controller: this
                }).then(function(oDialog) {
                    this._modelEvaluator = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadEvaluationData(aSelectedContexts);
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadEvaluationData(aSelectedContexts);
                this._modelEvaluator.open();
            }
        },

        // Benchmark Runner Action
        onBenchmarkRunner: function() {
            const oBinding = this.base.getView().byId("fe::table::EmbeddingModels::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectModelsFirst"));
                return;
            }

            if (!this._benchmarkRunner) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent14.ext.fragment.BenchmarkRunner",
                    controller: this
                }).then(function(oDialog) {
                    this._benchmarkRunner = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadBenchmarkSuites(aSelectedContexts);
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadBenchmarkSuites(aSelectedContexts);
                this._benchmarkRunner.open();
            }
        },

        // Hyperparameter Tuner Action
        onHyperparameterTuner: function() {
            const oBinding = this.base.getView().byId("fe::table::EmbeddingModels::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectModelsFirst"));
                return;
            }

            if (!this._hyperparameterTuner) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent14.ext.fragment.HyperparameterTuner",
                    controller: this
                }).then(function(oDialog) {
                    this._hyperparameterTuner = oDialog;
                    this.getView().addDependent(oDialog);
                    this._initializeHyperparameterTuner(aSelectedContexts[0]);
                    oDialog.open();
                }.bind(this));
            } else {
                this._initializeHyperparameterTuner(aSelectedContexts[0]);
                this._hyperparameterTuner.open();
            }
        },

        // Vector Optimizer Action
        onVectorOptimizer: function() {
            if (!this._vectorOptimizer) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent14.ext.fragment.VectorOptimizer",
                    controller: this
                }).then(function(oDialog) {
                    this._vectorOptimizer = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadVectorDatabaseData();
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadVectorDatabaseData();
                this._vectorOptimizer.open();
            }
        },

        // Performance Analyzer Action
        onPerformanceAnalyzer: function() {
            const oBinding = this.base.getView().byId("fe::table::EmbeddingModels::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectModelsFirst"));
                return;
            }

            if (!this._performanceAnalyzer) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent14.ext.fragment.PerformanceAnalyzer",
                    controller: this
                }).then(function(oDialog) {
                    this._performanceAnalyzer = oDialog;
                    this.getView().addDependent(oDialog);
                    this._analyzePerformance(aSelectedContexts);
                    oDialog.open();
                }.bind(this));
            } else {
                this._analyzePerformance(aSelectedContexts);
                this._performanceAnalyzer.open();
            }
        },

        // Real-time Updates via WebSocket
        onAfterRendering: function() {
            this._initializeWebSocket();
        },

        _initializeWebSocket: function() {
            if (this._ws) return;

            try {
                this._ws = SecurityUtils.createSecureWebSocket('wss://localhost:8014/embedding/updates', {
                    onmessage: function(event) {
                        const data = JSON.parse(event.data);
                        this._handleEmbeddingUpdate(data);
                    }.bind(this),
                    onerror: function(error) {
                        console.warn("Secure WebSocket error:", error);
                        this._initializePolling();
                    }.bind(this)
                });
                
                if (this._ws) {
                    this._ws.onclose = function() {
                        setTimeout(() => this._initializeWebSocket(), 5000);
                    }.bind(this);
                }

            } catch (error) {
                console.warn("WebSocket connection failed, falling back to polling");
                this._initializePolling();
            }
        },

        _initializePolling: function() {
            this._pollInterval = setInterval(() => {
                this._refreshModelData();
            }, 5000);
        },

        _handleEmbeddingUpdate: function(data) {
            const oModel = this.getView().getModel();
            
            switch (data.type) {
                case 'TRAINING_STARTED':
                    MessageToast.show(this.getResourceBundle().getText("msg.fineTuningStarted"));
                    break;
                case 'TRAINING_PROGRESS':
                    this._updateTrainingProgress(data);
                    break;
                case 'TRAINING_COMPLETED':
                    MessageToast.show(this.getResourceBundle().getText("msg.fineTuningCompleted"));
                    this._refreshModelData();
                    break;
                case 'TRAINING_FAILED':
                    MessageToast.show(this.getResourceBundle().getText("error.fineTuningFailed"));
                    break;
                case 'EVALUATION_COMPLETED':
                    MessageToast.show(this.getResourceBundle().getText("msg.evaluationCompleted"));
                    this._refreshModelData();
                    break;
                case 'BENCHMARK_UPDATE':
                    this._updateBenchmarkResults(data);
                    break;
            }
        },

        _loadDashboardData: function() {
            const oModel = this.getView().getModel();
            
            // Load embedding statistics
            SecurityUtils.secureCallFunction(oModel, "/GetEmbeddingStatistics", {
                success: function(data) {
                    this._updateDashboardCharts(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingStatistics"));
                }.bind(this)
            });
        },

        _initializeFineTuningWizard: function(oContext) {
            const oModel = this.getView().getModel();
            const sModelId = oContext.getObject().modelId;
            
            if (!SecurityUtils.checkEmbeddingAuth('GetModelConfiguration', {})) {
                return;
            }

            SecurityUtils.secureCallFunction(oModel, "/GetModelConfiguration", {
                urlParameters: {
                    modelId: sModelId
                },
                success: function(data) {
                    this._setupFineTuningWizard(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingModelConfig"));
                }.bind(this)
            });
        },

        _loadEvaluationData: function(aSelectedContexts) {
            const oModel = this.getView().getModel();
            const aModelIds = aSelectedContexts.map(ctx => ctx.getObject().modelId);
            
            SecurityUtils.secureCallFunction(oModel, "/GetEvaluationMetrics", {
                urlParameters: {
                    modelIds: aModelIds.join(',')
                },
                success: function(data) {
                    this._displayEvaluationResults(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingEvaluationData"));
                }.bind(this)
            });
        },

        _loadBenchmarkSuites: function(aSelectedContexts) {
            const oModel = this.getView().getModel();
            
            SecurityUtils.secureCallFunction(oModel, "/GetAvailableBenchmarks", {
                success: function(data) {
                    this._setupBenchmarkRunner(data, aSelectedContexts);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingBenchmarks"));
                }.bind(this)
            });
        },

        _initializeHyperparameterTuner: function(oContext) {
            const oModel = this.getView().getModel();
            const sModelId = oContext.getObject().modelId;
            
            if (!SecurityUtils.checkEmbeddingAuth('GetHyperparameterSpace', {})) {
                return;
            }

            SecurityUtils.secureCallFunction(oModel, "/GetHyperparameterSpace", {
                urlParameters: {
                    modelId: sModelId
                },
                success: function(data) {
                    this._setupHyperparameterTuner(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingHyperparameters"));
                }.bind(this)
            });
        },

        _loadVectorDatabaseData: function() {
            const oModel = this.getView().getModel();
            
            SecurityUtils.secureCallFunction(oModel, "/GetVectorDatabases", {
                success: function(data) {
                    this._updateVectorDatabaseList(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingVectorDatabases"));
                }.bind(this)
            });
        },

        _analyzePerformance: function(aSelectedContexts) {
            const oModel = this.getView().getModel();
            const aModelIds = aSelectedContexts.map(ctx => ctx.getObject().modelId);
            
            if (!SecurityUtils.checkEmbeddingAuth('AnalyzeModelPerformance', {})) {
                return;
            }

            SecurityUtils.secureCallFunction(oModel, "/AnalyzeModelPerformance", {
                urlParameters: {
                    modelIds: aModelIds.join(',')
                },
                success: function(data) {
                    this._displayPerformanceAnalysis(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.performanceAnalysisFailed"));
                }.bind(this)
            });
        },

        _refreshModelData: function() {
            const oBinding = this.base.getView().byId("fe::table::EmbeddingModels::LineItem").getBinding("rows");
            oBinding.refresh();
        },

        _updateTrainingProgress: function(data) {
            // Update training progress in real-time
        },

        _updateBenchmarkResults: function(data) {
            // Update benchmark results in real-time
        },

        _updateDashboardCharts: function(data) {
            // Update dashboard charts with statistics
        },

        _setupFineTuningWizard: function(data) {
            // Setup fine-tuning wizard with model configuration
        },

        _displayEvaluationResults: function(data) {
            // Display evaluation results in dialog
        },

        _setupBenchmarkRunner: function(data, contexts) {
            // Setup benchmark runner with available benchmarks
        },

        _setupHyperparameterTuner: function(data) {
            // Setup hyperparameter tuner with search space
        },

        _updateVectorDatabaseList: function(data) {
            // Update vector database list
        },

        _displayPerformanceAnalysis: function(data) {
            // Display performance analysis results
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