sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel",
    "a2a/ext/agent14/utils/SecurityUtils"
], function(ControllerExtension, MessageToast, MessageBox, Fragment, JSONModel, SecurityUtils) {
    "use strict";

    /**
     * @class a2a.network.agent14.ext.controller.ListReportExt
     * @extends sap.ui.core.mvc.ControllerExtension
     * @description Controller extension for Agent 14 List Report - Embedding Fine-Tuner Agent.
     * Provides comprehensive embedding model fine-tuning capabilities including hyperparameter optimization,
     * model evaluation, benchmark testing, and vector database optimization with enterprise-grade security.
     */
    return ControllerExtension.extend("a2a.network.agent14.ext.controller.ListReportExt", {
        
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
                this._startRealtimeEmbeddingUpdates();
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
            // Debounce model search operations
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
         * @description Performs search operation for embedding models.
         * @param {string} sQuery - Search query
         * @private
         */
        _performSearch: function(sQuery) {
            // Implement search logic for embedding models
        },

        /**
         * @function onFineTuningDashboard
         * @description Opens comprehensive fine-tuning analytics dashboard with training metrics and performance insights.
         * @public
         */
        onFineTuningDashboard: function() {
            this._getOrCreateDialog("fineTuningDashboard", "a2a.network.agent14.ext.fragment.FineTuningDashboard")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadDashboardData(oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Fine-Tuning Dashboard: " + error.message);
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
         * @function onCreateEmbeddingModel
         * @description Opens dialog to create new embedding model with configuration options.
         * @public
         */
        onCreateEmbeddingModel: function() {
            this._getOrCreateDialog("createModelDialog", "a2a.network.agent14.ext.fragment.CreateEmbeddingModel")
                .then(function(oDialog) {
                    var oModel = new JSONModel({
                        modelName: "",
                        description: "",
                        modelType: "sentence_bert",
                        baseModel: "bert_base",
                        embeddingDimension: 768,
                        architecture: "transformer",
                        tokenizer: "wordpiece",
                        normalization: true,
                        quantization: false,
                        mixedPrecision: true,
                        autoOptimization: true
                    });
                    oDialog.setModel(oModel, "create");
                    oDialog.open();
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Create Embedding Model dialog: " + error.message);
                });
        },

        /**
         * @function onStartFineTuning
         * @description Opens fine-tuning wizard for selected embedding model.
         * @public
         */
        onStartFineTuning: function() {
            const oBinding = this.base.getView().byId("fe::table::EmbeddingModels::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectModelsFirst"));
                return;
            }

            this._getOrCreateDialog("fineTuningWizard", "a2a.network.agent14.ext.fragment.FineTuningWizard")
                .then(function(oDialog) {
                    oDialog.open();
                    this._initializeFineTuningWizard(aSelectedContexts[0], oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Fine-Tuning Wizard: " + error.message);
                });
        },

        /**
         * @function onModelEvaluator
         * @description Opens model evaluation interface for performance assessment.
         * @public
         */
        onModelEvaluator: function() {
            const oBinding = this.base.getView().byId("fe::table::EmbeddingModels::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectModelsFirst"));
                return;
            }

            this._getOrCreateDialog("modelEvaluator", "a2a.network.agent14.ext.fragment.ModelEvaluator")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadEvaluationData(aSelectedContexts, oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Model Evaluator: " + error.message);
                });
        },

        /**
         * @function onBenchmarkRunner
         * @description Opens benchmark runner for model performance testing.
         * @public
         */
        onBenchmarkRunner: function() {
            const oBinding = this.base.getView().byId("fe::table::EmbeddingModels::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectModelsFirst"));
                return;
            }

            this._getOrCreateDialog("benchmarkRunner", "a2a.network.agent14.ext.fragment.BenchmarkRunner")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadBenchmarkSuites(aSelectedContexts, oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Benchmark Runner: " + error.message);
                });
        },

        /**
         * @function onHyperparameterTuner
         * @description Opens hyperparameter optimization interface for model tuning.
         * @public
         */
        onHyperparameterTuner: function() {
            const oBinding = this.base.getView().byId("fe::table::EmbeddingModels::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectModelsFirst"));
                return;
            }

            this._getOrCreateDialog("hyperparameterTuner", "a2a.network.agent14.ext.fragment.HyperparameterTuner")
                .then(function(oDialog) {
                    oDialog.open();
                    this._initializeHyperparameterTuner(aSelectedContexts[0], oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Hyperparameter Tuner: " + error.message);
                });
        },

        /**
         * @function onVectorOptimizer
         * @description Opens vector database optimization interface for embedding storage optimization.
         * @public
         */
        onVectorOptimizer: function() {
            this._getOrCreateDialog("vectorOptimizer", "a2a.network.agent14.ext.fragment.VectorOptimizer")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadVectorDatabaseData(oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Vector Optimizer: " + error.message);
                });
        },

        /**
         * @function onPerformanceAnalyzer
         * @description Opens performance analysis interface for model optimization insights.
         * @public
         */
        onPerformanceAnalyzer: function() {
            const oBinding = this.base.getView().byId("fe::table::EmbeddingModels::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectModelsFirst"));
                return;
            }

            this._getOrCreateDialog("performanceAnalyzer", "a2a.network.agent14.ext.fragment.PerformanceAnalyzer")
                .then(function(oDialog) {
                    oDialog.open();
                    this._analyzePerformance(aSelectedContexts, oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Performance Analyzer: " + error.message);
                });
        },

        /**
         * @function _startRealtimeEmbeddingUpdates
         * @description Starts real-time updates for training and evaluation events.
         * @private
         */
        _startRealtimeEmbeddingUpdates: function() {
            this._initializeWebSocket();
        },

        /**
         * @function _initializeWebSocket
         * @description Initializes secure WebSocket connection for real-time embedding updates.
         * @private
         */
        _initializeWebSocket: function() {
            if (this._ws) return;

            // Validate WebSocket URL for security
            const wsUrl = 'wss://localhost:8014/embedding/updates';
            if (!this._securityUtils.validateWebSocketUrl(wsUrl)) {
                MessageBox.error("Invalid WebSocket URL for security reasons");
                this._securityUtils.logSecureOperation('WEBSOCKET_VALIDATION_FAILED', 'ERROR', { url: wsUrl });
                return;
            }

            try {
                this._ws = SecurityUtils.createSecureWebSocket('wss://localhost:8014/embedding/updates', {
                    onmessage: function(event) {
                        try {
                            const data = JSON.parse(event.data);
                            this._handleEmbeddingUpdate(data);
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
                this._refreshModelData();
            }, 10000);
        },

        /**
         * @function _handleEmbeddingUpdate
         * @description Handles real-time embedding updates from WebSocket.
         * @param {Object} data - Update data
         * @private
         */
        _handleEmbeddingUpdate: function(data) {
            try {
                // Sanitize incoming data
                const sanitizedData = SecurityUtils.sanitizeEmbeddingData(JSON.stringify(data));
                const parsedData = JSON.parse(sanitizedData);
                
                var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                
                switch (parsedData.type) {
                    case 'TRAINING_STARTED':
                        var sTrainingStarted = oBundle.getText("msg.fineTuningStarted") || "Fine-tuning started";
                        MessageToast.show(sTrainingStarted);
                        break;
                    case 'TRAINING_PROGRESS':
                        this._updateTrainingProgress(parsedData);
                        break;
                    case 'TRAINING_COMPLETED':
                        var sTrainingCompleted = oBundle.getText("msg.fineTuningCompleted") || "Fine-tuning completed";
                        MessageToast.show(sTrainingCompleted);
                        this._refreshModelData();
                        break;
                    case 'TRAINING_FAILED':
                        var sTrainingFailed = oBundle.getText("error.fineTuningFailed") || "Fine-tuning failed";
                        MessageToast.show(sTrainingFailed);
                        break;
                    case 'EVALUATION_COMPLETED':
                        var sEvalCompleted = oBundle.getText("msg.evaluationCompleted") || "Evaluation completed";
                        MessageToast.show(sEvalCompleted);
                        this._refreshModelData();
                        break;
                    case 'BENCHMARK_UPDATE':
                        this._updateBenchmarkResults(parsedData);
                        break;
                    case 'OPTIMIZATION_UPDATE':
                        this._refreshModelData();
                        break;
                }
            } catch (error) {
                console.error("Error processing embedding update:", error);
            }
        },

        /**
         * @function _loadDashboardData
         * @description Loads fine-tuning dashboard data with statistics and training metrics.
         * @param {sap.m.Dialog} oDialog - Target dialog (optional)
         * @private
         */
        _loadDashboardData: function(oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["fineTuningDashboard"];
            if (!oTargetDialog) return;
            
            oTargetDialog.setBusy(true);
            
            // Check authorization before loading statistics
            if (!SecurityUtils.checkEmbeddingAuth('GetEmbeddingStatistics', {})) {
                oTargetDialog.setBusy(false);
                return;
            }

            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    var oModel = this.base.getView().getModel();
                    SecurityUtils.secureCallFunction(oModel, "/GetEmbeddingStatistics", {
                        success: function(data) {
                            // Validate and sanitize response data
                            if (this._validateStatisticsResponse(data)) {
                                resolve(data);
                            } else {
                                reject(new Error("Invalid statistics response format"));
                            }
                        }.bind(this),
                        error: function(error) {
                            var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                            var sErrorMsg = oBundle.getText("error.loadingStatistics") || "Error loading statistics";
                            SecurityUtils.logSecureOperation('STATISTICS_LOAD_ERROR', 'ERROR', { error: error });
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
         * @function _initializeFineTuningWizard
         * @description Initializes fine-tuning wizard with model configuration.
         * @param {sap.ui.model.Context} oContext - Selected model context
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _initializeFineTuningWizard: function(oContext, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["fineTuningWizard"];
            if (!oTargetDialog) return;
            
            oTargetDialog.setBusy(true);
            const sModelId = oContext.getObject().modelId;
            
            if (!SecurityUtils.checkEmbeddingAuth('GetModelConfiguration', {})) {
                return;
            }

            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    var oModel = this.base.getView().getModel();
                    SecurityUtils.secureCallFunction(oModel, "/GetModelConfiguration", {
                        urlParameters: {
                            modelId: sModelId
                        },
                        success: function(data) {
                            resolve(data);
                        },
                        error: function(error) {
                            var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                            var sErrorMsg = oBundle.getText("error.loadingModelConfig") || "Error loading model configuration";
                            reject(new Error(sErrorMsg));
                        }.bind(this)
                    });
                }.bind(this));
            }.bind(this)).then(function(data) {
                oTargetDialog.setBusy(false);
                this._setupFineTuningWizard(data, oTargetDialog);
            }.bind(this)).catch(function(error) {
                oTargetDialog.setBusy(false);
                MessageBox.error(error.message);
            });
        },

        /**
         * @function _loadEvaluationData
         * @description Loads evaluation metrics for selected models.
         * @param {Array<sap.ui.model.Context>} aSelectedContexts - Selected model contexts
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _loadEvaluationData: function(aSelectedContexts, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["modelEvaluator"];
            if (!oTargetDialog) return;
            
            oTargetDialog.setBusy(true);
            const aModelIds = aSelectedContexts.map(ctx => ctx.getObject().modelId);
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    var oModel = this.base.getView().getModel();
                    SecurityUtils.secureCallFunction(oModel, "/GetEvaluationMetrics", {
                        urlParameters: {
                            modelIds: aModelIds.join(',')
                        },
                        success: function(data) {
                            resolve(data);
                        },
                        error: function(error) {
                            var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                            var sErrorMsg = oBundle.getText("error.loadingEvaluationData") || "Error loading evaluation data";
                            reject(new Error(sErrorMsg));
                        }.bind(this)
                    });
                }.bind(this));
            }.bind(this)).then(function(data) {
                oTargetDialog.setBusy(false);
                this._displayEvaluationResults(data, oTargetDialog);
            }.bind(this)).catch(function(error) {
                oTargetDialog.setBusy(false);
                MessageBox.error(error.message);
            });
        },

        /**
         * @function _loadBenchmarkSuites
         * @description Loads available benchmark suites for testing.
         * @param {Array<sap.ui.model.Context>} aSelectedContexts - Selected model contexts
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _loadBenchmarkSuites: function(aSelectedContexts, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["benchmarkRunner"];
            if (!oTargetDialog) return;
            
            oTargetDialog.setBusy(true);
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    var oModel = this.base.getView().getModel();
                    SecurityUtils.secureCallFunction(oModel, "/GetAvailableBenchmarks", {
                        success: function(data) {
                            resolve(data);
                        },
                        error: function(error) {
                            var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                            var sErrorMsg = oBundle.getText("error.loadingBenchmarks") || "Error loading benchmarks";
                            reject(new Error(sErrorMsg));
                        }.bind(this)
                    });
                }.bind(this));
            }.bind(this)).then(function(data) {
                oTargetDialog.setBusy(false);
                this._setupBenchmarkRunner(data, aSelectedContexts, oTargetDialog);
            }.bind(this)).catch(function(error) {
                oTargetDialog.setBusy(false);
                MessageBox.error(error.message);
            });
        },

        /**
         * @function _initializeHyperparameterTuner
         * @description Initializes hyperparameter tuner with search space configuration.
         * @param {sap.ui.model.Context} oContext - Selected model context
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _initializeHyperparameterTuner: function(oContext, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["hyperparameterTuner"];
            if (!oTargetDialog) return;
            
            oTargetDialog.setBusy(true);
            const sModelId = oContext.getObject().modelId;
            
            if (!SecurityUtils.checkEmbeddingAuth('GetHyperparameterSpace', {})) {
                return;
            }

            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    var oModel = this.base.getView().getModel();
                    SecurityUtils.secureCallFunction(oModel, "/GetHyperparameterSpace", {
                        urlParameters: {
                            modelId: sModelId
                        },
                        success: function(data) {
                            resolve(data);
                        },
                        error: function(error) {
                            var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                            var sErrorMsg = oBundle.getText("error.loadingHyperparameters") || "Error loading hyperparameters";
                            reject(new Error(sErrorMsg));
                        }.bind(this)
                    });
                }.bind(this));
            }.bind(this)).then(function(data) {
                oTargetDialog.setBusy(false);
                this._setupHyperparameterTuner(data, oTargetDialog);
            }.bind(this)).catch(function(error) {
                oTargetDialog.setBusy(false);
                MessageBox.error(error.message);
            });
        },

        /**
         * @function _loadVectorDatabaseData
         * @description Loads vector database configurations and status.
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _loadVectorDatabaseData: function(oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["vectorOptimizer"];
            if (!oTargetDialog) return;
            
            oTargetDialog.setBusy(true);
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    var oModel = this.base.getView().getModel();
                    SecurityUtils.secureCallFunction(oModel, "/GetVectorDatabases", {
                        success: function(data) {
                            resolve(data);
                        },
                        error: function(error) {
                            var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                            var sErrorMsg = oBundle.getText("error.loadingVectorDatabases") || "Error loading vector databases";
                            reject(new Error(sErrorMsg));
                        }.bind(this)
                    });
                }.bind(this));
            }.bind(this)).then(function(data) {
                oTargetDialog.setBusy(false);
                this._updateVectorDatabaseList(data, oTargetDialog);
            }.bind(this)).catch(function(error) {
                oTargetDialog.setBusy(false);
                MessageBox.error(error.message);
            });
        },

        /**
         * @function _analyzePerformance
         * @description Analyzes performance metrics for selected models.
         * @param {Array<sap.ui.model.Context>} aSelectedContexts - Selected model contexts
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _analyzePerformance: function(aSelectedContexts, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["performanceAnalyzer"];
            if (!oTargetDialog) return;
            
            oTargetDialog.setBusy(true);
            const aModelIds = aSelectedContexts.map(ctx => ctx.getObject().modelId);
            
            if (!SecurityUtils.checkEmbeddingAuth('AnalyzeModelPerformance', {})) {
                return;
            }

            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    var oModel = this.base.getView().getModel();
                    SecurityUtils.secureCallFunction(oModel, "/AnalyzeModelPerformance", {
                        urlParameters: {
                            modelIds: aModelIds.join(',')
                        },
                        success: function(data) {
                            resolve(data);
                        },
                        error: function(error) {
                            var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                            var sErrorMsg = oBundle.getText("error.performanceAnalysisFailed") || "Performance analysis failed";
                            reject(new Error(sErrorMsg));
                        }.bind(this)
                    });
                }.bind(this));
            }.bind(this)).then(function(data) {
                oTargetDialog.setBusy(false);
                this._displayPerformanceAnalysis(data, oTargetDialog);
            }.bind(this)).catch(function(error) {
                oTargetDialog.setBusy(false);
                MessageBox.error(error.message);
            });
        },

        /**
         * @function _refreshModelData
         * @description Refreshes embedding model data in the list.
         * @private
         */
        _refreshModelData: function() {
            const oBinding = this.base.getView().byId("fe::table::EmbeddingModels::LineItem").getBinding("rows");
            if (oBinding) {
                oBinding.refresh();
            }
        },

        /**
         * @function _updateTrainingProgress
         * @description Updates training progress indicators in real-time.
         * @param {Object} data - Training progress data
         * @private
         */
        _updateTrainingProgress: function(data) {
            // Update progress indicators
            var oProgressDialog = this._dialogCache["fineTuningWizard"];
            if (oProgressDialog && oProgressDialog.isOpen()) {
                var oProgressIndicator = oProgressDialog.byId("trainingProgress");
                if (oProgressIndicator) {
                    oProgressIndicator.setPercentValue(data.progress);
                    oProgressIndicator.setDisplayValue(`${data.progress}% - Epoch ${data.epoch}/${data.totalEpochs}`);
                }
                
                // Update loss chart if available
                if (data.trainingLoss && data.validationLoss) {
                    this._updateLossChart(data, oProgressDialog);
                }
            }
        },

        /**
         * @function _updateBenchmarkResults
         * @description Updates benchmark results in real-time.
         * @param {Object} data - Benchmark results data
         * @private
         */
        _updateBenchmarkResults: function(data) {
            // Update benchmark results in UI
            var oBenchmarkDialog = this._dialogCache["benchmarkRunner"];
            if (oBenchmarkDialog && oBenchmarkDialog.isOpen()) {
                var oBenchmarkModel = oBenchmarkDialog.getModel("benchmark");
                if (oBenchmarkModel) {
                    var oData = oBenchmarkModel.getData();
                    if (!oData.results) oData.results = [];
                    oData.results.push(data.result);
                    oBenchmarkModel.setData(oData);
                }
            }
        },

        /**
         * @function _updateDashboardCharts
         * @description Updates fine-tuning dashboard charts with embedding statistics.
         * @param {Object} data - Dashboard data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _updateDashboardCharts: function(data, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["fineTuningDashboard"];
            if (!oTargetDialog) return;
            
            this._createTrainingProgressChart(data.trainingProgress, oTargetDialog);
            this._createLossConvergenceChart(data.lossConvergence, oTargetDialog);
            this._createPerformanceMetricsChart(data.performanceMetrics, oTargetDialog);
            this._createEmbeddingDistributionChart(data.embeddingDistribution, oTargetDialog);
        },

        /**
         * @function _setupFineTuningWizard
         * @description Sets up fine-tuning wizard with model configuration.
         * @param {Object} data - Model configuration data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _setupFineTuningWizard: function(data, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["fineTuningWizard"];
            if (!oTargetDialog) return;
            
            var oWizardModel = new JSONModel({
                modelConfig: data.configuration,
                trainingSettings: {
                    learningRate: data.configuration.learningRate || 0.001,
                    batchSize: data.configuration.batchSize || 32,
                    epochs: data.configuration.epochs || 10,
                    optimizer: data.configuration.optimizer || "adamw",
                    lossFunction: data.configuration.lossFunction || "cosine",
                    warmupSteps: data.configuration.warmupSteps || 100,
                    gradientClipping: data.configuration.gradientClipping || 1.0
                },
                datasetOptions: data.datasetOptions,
                augmentationOptions: data.augmentationOptions
            });
            oTargetDialog.setModel(oWizardModel, "wizard");
        },

        /**
         * @function _displayEvaluationResults
         * @description Displays model evaluation results in dialog.
         * @param {Object} data - Evaluation results data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _displayEvaluationResults: function(data, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["modelEvaluator"];
            if (!oTargetDialog) return;
            
            var oEvalModel = new JSONModel({
                evaluationResults: data.results,
                comparisonData: data.comparison,
                metrics: data.metrics,
                recommendations: data.recommendations
            });
            oTargetDialog.setModel(oEvalModel, "evaluation");
        },

        /**
         * @function _setupBenchmarkRunner
         * @description Sets up benchmark runner with available test suites.
         * @param {Object} data - Benchmark suites data
         * @param {Array<sap.ui.model.Context>} contexts - Selected model contexts
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _setupBenchmarkRunner: function(data, contexts, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["benchmarkRunner"];
            if (!oTargetDialog) return;
            
            var oBenchmarkModel = new JSONModel({
                benchmarkSuites: data.suites,
                selectedModels: contexts.map(ctx => ctx.getObject()),
                configuration: {
                    parallel: true,
                    compareWithBaseline: true,
                    generateReport: true
                },
                results: []
            });
            oTargetDialog.setModel(oBenchmarkModel, "benchmark");
        },

        /**
         * @function _setupHyperparameterTuner
         * @description Sets up hyperparameter tuner with search space configuration.
         * @param {Object} data - Hyperparameter space data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _setupHyperparameterTuner: function(data, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["hyperparameterTuner"];
            if (!oTargetDialog) return;
            
            var oHyperModel = new JSONModel({
                searchSpace: data.searchSpace,
                optimizationMethods: data.methods,
                currentConfig: data.currentConfig,
                searchHistory: data.history,
                bestConfig: data.bestConfig
            });
            oTargetDialog.setModel(oHyperModel, "hyperparameter");
        },

        /**
         * @function _updateVectorDatabaseList
         * @description Updates vector database list and optimization options.
         * @param {Object} data - Vector database data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _updateVectorDatabaseList: function(data, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["vectorOptimizer"];
            if (!oTargetDialog) return;
            
            var oVectorModel = new JSONModel({
                databases: data.databases,
                indexTypes: data.indexTypes,
                optimizationStrategies: data.strategies,
                performanceMetrics: data.metrics
            });
            oTargetDialog.setModel(oVectorModel, "vector");
        },

        /**
         * @function _displayPerformanceAnalysis
         * @description Displays model performance analysis results.
         * @param {Object} data - Performance analysis data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _displayPerformanceAnalysis: function(data, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["performanceAnalyzer"];
            if (!oTargetDialog) return;
            
            var oPerfModel = new JSONModel({
                performanceMetrics: data.metrics,
                resourceUtilization: data.resources,
                optimizationSuggestions: data.suggestions,
                benchmarkComparison: data.comparison
            });
            oTargetDialog.setModel(oPerfModel, "performance");
            
            // Create performance charts
            this._createResourceUtilizationChart(data.resources, oTargetDialog);
            this._createInferenceSpeedChart(data.metrics.inferenceSpeed, oTargetDialog);
        },

        /**
         * @function _createTrainingProgressChart
         * @description Creates training progress chart for dashboard.
         * @param {Object} data - Chart data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createTrainingProgressChart: function(data, oDialog) {
            var oChartContainer = oDialog.byId("trainingProgressChart");
            if (!oChartContainer || !data) return;
            
            var oChartModel = new JSONModel({
                chartData: data,
                config: {
                    title: this.getResourceBundle().getText("chart.trainingProgress"),
                    xAxisLabel: this.getResourceBundle().getText("field.epoch"),
                    yAxisLabel: this.getResourceBundle().getText("field.loss")
                }
            });
            oChartContainer.setModel(oChartModel, "chart");
        },
        
        /**
         * @function _createLossConvergenceChart
         * @description Creates loss convergence chart for training monitoring.
         * @param {Object} data - Chart data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createLossConvergenceChart: function(data, oDialog) {
            var oChartContainer = oDialog.byId("lossConvergenceChart");
            if (!oChartContainer || !data) return;
            
            var oChartModel = new JSONModel({
                chartData: data,
                config: {
                    title: this.getResourceBundle().getText("chart.lossConvergence"),
                    showLegend: true,
                    series: ["Training Loss", "Validation Loss"]
                }
            });
            oChartContainer.setModel(oChartModel, "chart");
        },
        
        /**
         * @function _createPerformanceMetricsChart
         * @description Creates performance metrics chart for model comparison.
         * @param {Object} data - Chart data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createPerformanceMetricsChart: function(data, oDialog) {
            var oChartContainer = oDialog.byId("performanceMetricsChart");
            if (!oChartContainer || !data) return;
            
            var oChartModel = new JSONModel({
                chartData: data,
                config: {
                    title: this.getResourceBundle().getText("chart.performanceMetrics"),
                    showDataLabels: true,
                    colorPalette: ["#5cbae6", "#b6d7a8", "#ffd93d", "#ff7b7b"]
                }
            });
            oChartContainer.setModel(oChartModel, "chart");
        },
        
        /**
         * @function _createEmbeddingDistributionChart
         * @description Creates embedding distribution visualization chart.
         * @param {Object} data - Chart data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createEmbeddingDistributionChart: function(data, oDialog) {
            var oChartContainer = oDialog.byId("embeddingDistributionChart");
            if (!oChartContainer || !data) return;
            
            var oChartModel = new JSONModel({
                chartData: data,
                config: {
                    title: this.getResourceBundle().getText("chart.embeddingDistribution"),
                    enableDrillDown: true,
                    visualizationType: "scatter"
                }
            });
            oChartContainer.setModel(oChartModel, "chart");
        },
        
        /**
         * @function _updateLossChart
         * @description Updates loss chart with real-time training data.
         * @param {Object} data - Loss data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _updateLossChart: function(data, oDialog) {
            var oChartContainer = oDialog.byId("lossChart");
            if (!oChartContainer) return;
            
            var oChartModel = oChartContainer.getModel("chart");
            if (oChartModel) {
                var oData = oChartModel.getData();
                if (!oData.series) {
                    oData.series = {
                        training: [],
                        validation: []
                    };
                }
                oData.series.training.push({x: data.epoch, y: data.trainingLoss});
                oData.series.validation.push({x: data.epoch, y: data.validationLoss});
                oChartModel.setData(oData);
            }
        },
        
        /**
         * @function _createResourceUtilizationChart
         * @description Creates resource utilization chart for performance analysis.
         * @param {Object} data - Resource data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createResourceUtilizationChart: function(data, oDialog) {
            var oChartContainer = oDialog.byId("resourceUtilizationChart");
            if (!oChartContainer || !data) return;
            
            var oChartModel = new JSONModel({
                chartData: data,
                config: {
                    title: this.getResourceBundle().getText("chart.resourceUtilization"),
                    type: "gauge",
                    maxValue: 100
                }
            });
            oChartContainer.setModel(oChartModel, "chart");
        },
        
        /**
         * @function _createInferenceSpeedChart
         * @description Creates inference speed comparison chart.
         * @param {Object} data - Speed data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createInferenceSpeedChart: function(data, oDialog) {
            var oChartContainer = oDialog.byId("inferenceSpeedChart");
            if (!oChartContainer || !data) return;
            
            var oChartModel = new JSONModel({
                chartData: data,
                config: {
                    title: this.getResourceBundle().getText("field.inferenceSpeed"),
                    yAxisLabel: this.getResourceBundle().getText("unit.vectorsPerSecond"),
                    type: "bar"
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
        },

        /**
         * @function _validateStatisticsResponse
         * @description Validates statistics response data for security
         * @param {object} data - Response data to validate
         * @returns {boolean} True if valid
         * @private
         */
        _validateStatisticsResponse: function(data) {
            if (!data || typeof data !== 'object') {
                return false;
            }
            
            // Check for required properties and reasonable values
            const requiredProps = ['trainingProgress', 'lossConvergence', 'performanceMetrics'];
            for (const prop of requiredProps) {
                if (!data[prop]) {
                    SecurityUtils.logSecureOperation('INVALID_STATISTICS_RESPONSE', 'WARNING', { 
                        missing: prop 
                    });
                    return false;
                }
            }
            
            // Validate numeric ranges
            if (data.performanceMetrics && data.performanceMetrics.accuracy) {
                const accuracy = parseFloat(data.performanceMetrics.accuracy);
                if (isNaN(accuracy) || accuracy < 0 || accuracy > 1) {
                    SecurityUtils.logSecureOperation('SUSPICIOUS_ACCURACY_VALUE', 'WARNING', { 
                        accuracy: accuracy 
                    });
                    return false;
                }
            }
            
            return true;
        },

        /**
         * @function _validateModelResponse
         * @description Validates model response data for security
         * @param {object} data - Response data to validate
         * @returns {boolean} True if valid
         * @private
         */
        _validateModelResponse: function(data) {
            if (!data || typeof data !== 'object') {
                return false;
            }
            
            // Sanitize model configuration data
            if (data.configuration) {
                const validation = SecurityUtils.validateHyperparameters(data.configuration);
                if (!validation.isValid) {
                    SecurityUtils.logSecureOperation('INVALID_MODEL_CONFIG', 'ERROR', { 
                        errors: validation.errors 
                    });
                    return false;
                }
            }
            
            return true;
        },

        /**
         * @function _sanitizeTrainingData
         * @description Sanitizes and validates training data
         * @param {object} trainingData - Training data to sanitize
         * @returns {object} Sanitized training data
         * @private
         */
        _sanitizeTrainingData: function(trainingData) {
            if (!trainingData) return {};
            
            const validation = SecurityUtils.validateTrainingData(trainingData);
            if (!validation.isValid) {
                SecurityUtils.logSecureOperation('TRAINING_DATA_VALIDATION_FAILED', 'ERROR', {
                    errors: validation.errors,
                    riskScore: validation.riskScore
                });
                return {};
            }
            
            return validation.sanitized;
        },

        /**
         * @function _validateModelPath
         * @description Validates model path for security before operations
         * @param {string} modelPath - Model path to validate
         * @returns {boolean} True if valid
         * @private
         */
        _validateModelPath: function(modelPath) {
            const validation = SecurityUtils.validateModelPath(modelPath);
            if (!validation.isValid) {
                SecurityUtils.logSecureOperation('MODEL_PATH_VALIDATION_FAILED', 'ERROR', {
                    path: modelPath,
                    errors: validation.errors,
                    riskScore: validation.riskScore
                });
                return false;
            }
            return true;
        },

        /**
         * @function _secureModelOperation
         * @description Wrapper for secure model operations
         * @param {string} operation - Operation name
         * @param {object} params - Operation parameters
         * @param {function} callback - Success callback
         * @param {function} errorCallback - Error callback
         * @private
         */
        _secureModelOperation: function(operation, params, callback, errorCallback) {
            // Check authorization
            if (!SecurityUtils.checkEmbeddingAuth(operation, params)) {
                if (errorCallback) {
                    errorCallback(new Error('Insufficient permissions'));
                }
                return;
            }
            
            // Validate and sanitize parameters
            const sanitizedParams = this._sanitizeOperationParams(params);
            
            // Log operation attempt
            SecurityUtils.logSecureOperation('MODEL_OPERATION_ATTEMPT', 'INFO', {
                operation: operation,
                params: Object.keys(sanitizedParams)
            });
            
            // Execute with enhanced error handling
            try {
                var oModel = this.base.getView().getModel();
                SecurityUtils.secureCallFunction(oModel, operation, {
                    urlParameters: sanitizedParams,
                    success: function(data) {
                        SecurityUtils.logSecureOperation('MODEL_OPERATION_SUCCESS', 'INFO', {
                            operation: operation
                        });
                        if (callback) callback(data);
                    },
                    error: function(error) {
                        SecurityUtils.logSecureOperation('MODEL_OPERATION_ERROR', 'ERROR', {
                            operation: operation,
                            error: error
                        });
                        if (errorCallback) errorCallback(error);
                    }
                });
            } catch (error) {
                SecurityUtils.logSecureOperation('MODEL_OPERATION_EXCEPTION', 'ERROR', {
                    operation: operation,
                    error: error
                });
                if (errorCallback) errorCallback(error);
            }
        },

        /**
         * @function _sanitizeOperationParams
         * @description Sanitizes operation parameters for security
         * @param {object} params - Parameters to sanitize
         * @returns {object} Sanitized parameters
         * @private
         */
        _sanitizeOperationParams: function(params) {
            if (!params || typeof params !== 'object') {
                return {};
            }
            
            const sanitized = {};
            for (const [key, value] of Object.entries(params)) {
                const cleanKey = SecurityUtils.sanitizeInput(key);
                if (cleanKey && typeof value === 'string') {
                    sanitized[cleanKey] = SecurityUtils.sanitizeInput(value);
                } else if (cleanKey && typeof value === 'number' && isFinite(value)) {
                    sanitized[cleanKey] = value;
                } else if (cleanKey && typeof value === 'boolean') {
                    sanitized[cleanKey] = value;
                }
            }
            
            return sanitized;
        }
    });
});