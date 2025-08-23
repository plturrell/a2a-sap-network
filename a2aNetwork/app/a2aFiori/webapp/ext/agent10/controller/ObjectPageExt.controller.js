sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel",
    "a2a/network/agent10/ext/utils/SecurityUtils"
], function(ControllerExtension, MessageToast, MessageBox, Fragment, JSONModel, SecurityUtils) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent10.ext.controller.ObjectPageExt", {

        override: {
            onInit: function () {
                this._initializeCreateModel();
            }
        },

        _initializeCreateModel: function() {
            var oCreateData = {
                taskName: "",
                description: "",
                calculationType: "",
                formulaCategory: "",
                priority: "medium",
                taskNameState: "",
                taskNameStateText: "",
                calculationTypeState: "",
                calculationTypeStateText: "",
                formulaCategoryState: "",
                formulaCategoryStateText: "",
                formulaExpression: "",
                formulaLanguage: "javascript",
                expectedDataType: "number",
                formulaComplexity: "medium",
                formulaSource: "custom",
                precisionLevel: "double",
                calculationEngine: "numpy",
                parallelProcessing: false,
                threadCount: 4,
                gpuAcceleration: false,
                cachingEnabled: true,
                resultValidation: true,
                confidenceInterval: 95,
                selfHealingEnabled: true
            };
            var oCreateModel = new JSONModel(oCreateData);
            this.getView().setModel(oCreateModel, "create");
        },

        onCreateCalculationTask: function() {
            var oView = this.getView();
            
            if (!this._oCreateDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent10.ext.fragment.CreateCalculationTask",
                    controller: this
                }).then(function(oDialog) {
                    this._oCreateDialog = oDialog;
                    oView.addDependent(this._oCreateDialog);
                    this._oCreateDialog.open();
                }.bind(this));
            } else {
                this._oCreateDialog.open();
            }
        },

        onTaskNameChange: function(oEvent) {
            var sValue = oEvent.getParameter("value");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            if (!sValue || sValue.length < 3) {
                oData.taskNameState = "Error";
                oData.taskNameStateText = "Task name must be at least 3 characters";
            } else if (sValue.length > 100) {
                oData.taskNameState = "Error";
                oData.taskNameStateText = "Task name must not exceed 100 characters";
            } else {
                oData.taskNameState = "Success";
                oData.taskNameStateText = "Valid task name";
            }
            
            oCreateModel.setData(oData);
        },

        onCalculationTypeChange: function(oEvent) {
            var sValue = oEvent.getParameter("selectedItem").getKey();
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            if (sValue) {
                oData.calculationTypeState = "Success";
                oData.calculationTypeStateText = "Calculation type selected";
                
                // Smart suggestions based on calculation type
                switch (sValue) {
                    case "basic":
                        oData.calculationEngine = "numpy";
                        oData.precisionLevel = "double";
                        oData.parallelProcessing = false;
                        break;
                    case "advanced":
                    case "scientific":
                        oData.calculationEngine = "scipy";
                        oData.precisionLevel = "extended";
                        oData.parallelProcessing = true;
                        oData.threadCount = 8;
                        break;
                    case "statistical":
                        oData.calculationEngine = "scipy";
                        oData.formulaCategory = "statistical";
                        oData.resultValidation = true;
                        break;
                    case "financial":
                        oData.calculationEngine = "numpy";
                        oData.formulaCategory = "financial";
                        oData.precisionLevel = "decimal128";
                        break;
                    case "matrix_operations":
                        oData.calculationEngine = "numpy";
                        oData.gpuAcceleration = true;
                        oData.parallelProcessing = true;
                        break;
                    case "optimization":
                        oData.calculationEngine = "scipy";
                        oData.precisionLevel = "quadruple";
                        oData.parallelProcessing = true;
                        break;
                }
            } else {
                oData.calculationTypeState = "Error";
                oData.calculationTypeStateText = "Please select a calculation type";
            }
            
            oCreateModel.setData(oData);
        },

        onFormulaCategoryChange: function(oEvent) {
            var sValue = oEvent.getParameter("selectedItem").getKey();
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            if (sValue) {
                oData.formulaCategoryState = "Success";
                oData.formulaCategoryStateText = "Formula category selected";
                
                // Adjust engine based on formula category
                switch (sValue) {
                    case "trigonometric":
                    case "logarithmic":
                    case "exponential":
                        oData.calculationEngine = "numpy";
                        oData.precisionLevel = "extended";
                        break;
                    case "statistical":
                        oData.calculationEngine = "scipy";
                        oData.resultValidation = true;
                        oData.confidenceInterval = 95;
                        break;
                    case "financial":
                        oData.precisionLevel = "decimal128";
                        oData.resultValidation = true;
                        break;
                    case "custom":
                        oData.calculationEngine = "custom";
                        oData.selfHealingEnabled = true;
                        break;
                }
            } else {
                oData.formulaCategoryState = "Error";
                oData.formulaCategoryStateText = "Please select a formula category";
            }
            
            oCreateModel.setData(oData);
        },

        onPriorityChange: function(oEvent) {
            var sValue = oEvent.getParameter("item").getKey();
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            // Adjust resource allocation based on priority
            switch (sValue) {
                case "high":
                    oData.threadCount = Math.min(16, oData.threadCount * 2);
                    oData.parallelProcessing = true;
                    oData.cachingEnabled = true;
                    break;
                case "medium":
                    oData.threadCount = 4;
                    break;
                case "low":
                    oData.threadCount = Math.max(1, Math.floor(oData.threadCount / 2));
                    oData.parallelProcessing = false;
                    break;
            }
            
            oCreateModel.setData(oData);
        },

        onCancelCreateCalculationTask: function() {
            this._oCreateDialog.close();
            this._resetCreateModel();
        },

        onConfirmCreateCalculationTask: function() {
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            // Final validation
            if (!this._validateCreateData(oData)) {
                return;
            }
            
            this._oCreateDialog.setBusy(true);
            
            // Sanitize data for security
            var oSanitizedData = {
                taskName: SecurityUtils.sanitizeInput(oData.taskName),
                description: SecurityUtils.sanitizeInput(oData.description),
                calculationType: oData.calculationType,
                formulaCategory: oData.formulaCategory,
                priority: oData.priority,
                precisionLevel: oData.precisionLevel,
                calculationEngine: oData.calculationEngine,
                parallelProcessing: !!oData.parallelProcessing,
                threadCount: parseInt(oData.threadCount) || 4,
                gpuAcceleration: !!oData.gpuAcceleration,
                cachingEnabled: !!oData.cachingEnabled,
                resultValidation: !!oData.resultValidation,
                confidenceInterval: parseFloat(oData.confidenceInterval) || 95,
                selfHealingEnabled: !!oData.selfHealingEnabled
            };
            
            SecurityUtils.secureCallFunction(this.getView().getModel(), "/CreateCalculationTask", {
                urlParameters: oSanitizedData,
                success: function(data) {
                    this._oCreateDialog.setBusy(false);
                    this._oCreateDialog.close();
                    MessageToast.show(this.getResourceBundle().getText("msg.calculationTaskCreated"));
                    this._refreshTaskData();
                    this._resetCreateModel();
                }.bind(this),
                error: function(error) {
                    this._oCreateDialog.setBusy(false);
                    var errorMsg = SecurityUtils.escapeHTML(error.message || "Unknown error");
                    MessageBox.error(this.getResourceBundle().getText("error.createTaskFailed") + ": " + errorMsg);
                }.bind(this)
            });
        },

        _validateCreateData: function(oData) {
            if (!oData.taskName || oData.taskName.length < 3) {
                MessageBox.error(this.getResourceBundle().getText("validation.taskNameRequired"));
                return false;
            }
            
            if (!oData.calculationType) {
                MessageBox.error(this.getResourceBundle().getText("validation.calculationTypeRequired"));
                return false;
            }
            
            if (!oData.formulaCategory) {
                MessageBox.error(this.getResourceBundle().getText("validation.formulaCategoryRequired"));
                return false;
            }
            
            return true;
        },

        _resetCreateModel: function() {
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.taskName = "";
            oData.description = "";
            oData.calculationType = "";
            oData.formulaCategory = "";
            oData.priority = "medium";
            oData.taskNameState = "";
            oData.taskNameStateText = "";
            oData.calculationTypeState = "";
            oData.calculationTypeStateText = "";
            oData.formulaCategoryState = "";
            oData.formulaCategoryStateText = "";
            oData.formulaExpression = "";
            oData.formulaLanguage = "javascript";
            oData.expectedDataType = "number";
            oData.formulaComplexity = "medium";
            oData.formulaSource = "custom";
            
            oCreateModel.setData(oData);
        },

        onFormulaChange: function(oEvent) {
            var sValue = oEvent.getParameter("value");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            // Basic formula validation
            if (sValue && sValue.length > 0) {
                // Check for basic syntax errors
                try {
                    // Simple validation - actual validation would be more complex
                    if (oData.formulaLanguage === "javascript") {
                        // Check for balanced parentheses and brackets
                        var openParen = (sValue.match(/\(/g) || []).length;
                        var closeParen = (sValue.match(/\)/g) || []).length;
                        var openBracket = (sValue.match(/\[/g) || []).length;
                        var closeBracket = (sValue.match(/\]/g) || []).length;
                        var openBrace = (sValue.match(/\{/g) || []).length;
                        var closeBrace = (sValue.match(/\}/g) || []).length;
                        
                        if (openParen !== closeParen || openBracket !== closeBracket || openBrace !== closeBrace) {
                            MessageToast.show("Warning: Unbalanced parentheses or brackets");
                        }
                    }
                } catch (e) {
                    MessageToast.show("Formula syntax error");
                }
            }
            
            oCreateModel.setData(oData);
        },

        onFormulaLanguageChange: function(oEvent) {
            var sValue = oEvent.getParameter("selectedItem").getKey();
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            // Update code editor type based on language
            var oCodeEditor = this.getView().byId("formulaEditor");
            if (oCodeEditor) {
                switch (sValue) {
                    case "python":
                        oCodeEditor.setType("python");
                        break;
                    case "r":
                        oCodeEditor.setType("r");
                        break;
                    case "sql":
                        oCodeEditor.setType("sql");
                        break;
                    default:
                        oCodeEditor.setType("javascript");
                }
            }
            
            oCreateModel.setData(oData);
        },
        

        // Execute Calculation Action
        onExecuteCalculation: function() {
            SecurityUtils.checkCalculationAuth('execute').then(function(authorized) {
                if (!authorized) {
                    MessageToast.show(this.getResourceBundle().getText("error.notAuthorized"));
                    return;
                }
                
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
            }.bind(this));
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
                this._eventSource = SecurityUtils.createSecureEventSource(`http://localhost:8010/calculations/${taskId}/stream`);
                
                this._eventSource.addEventListener('calculation-progress', (event) => {
                    this._updateCalculationProgress(event.data);
                });

                this._eventSource.addEventListener('calculation-completed', (event) => {
                    this._handleCalculationCompleted(event.data);
                });

                this._eventSource.addEventListener('calculation-error', (event) => {
                    this._handleCalculationError(event.data);
                });

                this._eventSource.addEventListener('self-healing', (event) => {
                    this._handleSelfHealingUpdate(event.data);
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
            
            SecurityUtils.secureCallFunction(oModel, "/ExecuteCalculation", {
                urlParameters: {
                    taskId: sTaskId
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.calculationExecuted"));
                    this._refreshTaskData();
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.calculationFailed", [SecurityUtils.escapeHTML(error.message)]));
                }.bind(this)
            });
        },

        _validateCalculationResult: function(oContext) {
            const oModel = this.getView().getModel();
            const sTaskId = oContext.getObject().taskId;
            
            SecurityUtils.secureCallFunction(oModel, "/ValidateResult", {
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
            
            SecurityUtils.secureCallFunction(oModel, "/TestPrecision", {
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
            
            SecurityUtils.secureCallFunction(oModel, "/RunSelfHealing", {
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
            
            SecurityUtils.secureCallFunction(oModel, "/GetOptimizationSuggestions", {
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
            
            SecurityUtils.secureCallFunction(oModel, "/GetPerformanceAnalysis", {
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
            
            SecurityUtils.secureCallFunction(oModel, "/GetVisualizationData", {
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
            MessageBox.error(this.getResourceBundle().getText("error.calculationFailed", [SecurityUtils.escapeHTML(data.error)]));
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