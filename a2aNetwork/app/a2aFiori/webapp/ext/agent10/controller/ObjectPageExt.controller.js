sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel",
    "a2a/network/agent10/ext/utils/SecurityUtils"
], (ControllerExtension, MessageToast, MessageBox, Fragment, JSONModel, SecurityUtils) => {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent10.ext.controller.ObjectPageExt", {

        override: {
            onInit() {
                this._extensionAPI = this.base.getExtensionAPI();
                this._securityUtils = SecurityUtils;
                this._initializeCreateModel();
                this._initializeSecurity();

                // Initialize device model for responsive behavior
                const oDeviceModel = new JSONModel(sap.ui.Device);
                oDeviceModel.setDefaultBindingMode(sap.ui.model.BindingMode.OneWay);
                this.base.getView().setModel(oDeviceModel, "device");
            },

            onExit() {
                this._cleanupResources();
                if (this.base.onExit) {
                    this.base.onExit.apply(this, arguments);
                }
            }
        },

        /**
         * @function onViewRealTimeMetrics
         * @description Opens real-time metrics monitoring dashboard with live data streams.
         * @public
         */
        onViewRealTimeMetrics() {
            if (!this._hasRole("MonitoringUser")) {
                MessageBox.error("Access denied. Monitoring User role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "ViewRealTimeMetrics", reason: "Insufficient permissions" });
                return;
            }

            const oContext = this.base.getView().getBindingContext();
            const sTaskId = oContext.getObject().taskId;
            const sTaskName = oContext.getObject().taskName;

            this._auditLogger.log("VIEW_REALTIME_METRICS", { taskId: sTaskId, taskName: sTaskName });

            this._getOrCreateDialog("realtimeMetrics", "a2a.network.agent10.ext.fragment.RealtimeMetrics")
                .then((oDialog) => {
                    oDialog.open();
                    this._startMetricsMonitoring(sTaskId, oDialog);
                })
                .catch((error) => {
                    MessageBox.error(`Failed to open Real-time Metrics: ${ error.message}`);
                });
        },

        /**
         * @function onExportMetrics
         * @description Exports monitoring metrics and reports in various formats.
         * @public
         */
        onExportMetrics() {
            if (!this._hasRole("MonitoringUser")) {
                MessageBox.error("Access denied. Monitoring User role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "ExportMetrics", reason: "Insufficient permissions" });
                return;
            }

            const oContext = this.base.getView().getBindingContext();
            const sTaskId = oContext.getObject().taskId;
            const sTaskName = oContext.getObject().taskName;

            this._auditLogger.log("EXPORT_METRICS", { taskId: sTaskId, taskName: sTaskName });

            this._getOrCreateDialog("exportMetrics", "a2a.network.agent10.ext.fragment.ExportMetrics")
                .then((oDialog) => {
                    const oExportModel = new JSONModel({
                        taskId: sTaskId,
                        taskName: sTaskName,
                        exportFormat: "JSON",
                        includeCharts: true,
                        includeRawData: true,
                        includeSummary: true,
                        dateRange: "ALL",
                        startDate: null,
                        endDate: null,
                        compressionEnabled: false
                    });
                    oDialog.setModel(oExportModel, "export");
                    oDialog.open();
                })
                .catch((error) => {
                    MessageBox.error(`Failed to open Export Metrics: ${ error.message}`);
                });
        },

        /**
         * @function onSetupAlerts
         * @description Opens alert setup interface for configuring monitoring thresholds.
         * @public
         */
        onSetupAlerts() {
            if (!this._hasRole("MonitoringAdmin")) {
                MessageBox.error("Access denied. Monitoring Administrator role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "SetupAlerts", reason: "Insufficient permissions" });
                return;
            }

            const oContext = this.base.getView().getBindingContext();
            const sTaskId = oContext.getObject().taskId;
            const sTaskName = oContext.getObject().taskName;

            this._auditLogger.log("SETUP_ALERTS", { taskId: sTaskId, taskName: sTaskName });

            this._getOrCreateDialog("setupAlerts", "a2a.network.agent10.ext.fragment.SetupAlerts")
                .then((oDialog) => {
                    oDialog.open();
                    this._loadAlertSetupData(sTaskId, oDialog);
                })
                .catch((error) => {
                    MessageBox.error(`Failed to open Setup Alerts: ${ error.message}`);
                });
        },

        /**
         * @function onViewLogs
         * @description Opens monitoring logs viewer with filtering and search capabilities.
         * @public
         */
        onViewLogs() {
            if (!this._hasRole("MonitoringUser")) {
                MessageBox.error("Access denied. Monitoring User role required.");
                this._auditLogger.log("ACCESS_DENIED", { action: "ViewLogs", reason: "Insufficient permissions" });
                return;
            }

            const oContext = this.base.getView().getBindingContext();
            const sTaskId = oContext.getObject().taskId;
            const sTaskName = oContext.getObject().taskName;

            this._auditLogger.log("VIEW_LOGS", { taskId: sTaskId, taskName: sTaskName });

            this._getOrCreateDialog("viewLogs", "a2a.network.agent10.ext.fragment.ViewLogs")
                .then((oDialog) => {
                    const oLogsModel = new JSONModel({
                        taskId: sTaskId,
                        taskName: sTaskName,
                        logLevel: "INFO",
                        dateRange: "TODAY",
                        startDate: new Date(),
                        endDate: new Date(),
                        searchQuery: "",
                        autoRefresh: true,
                        refreshInterval: 5000
                    });
                    oDialog.setModel(oLogsModel, "logs");
                    oDialog.open();
                    this._loadMonitoringLogs(sTaskId, oDialog);
                })
                .catch((error) => {
                    MessageBox.error(`Failed to open View Logs: ${ error.message}`);
                });
        },

        _initializeCreateModel() {
            const oCreateData = {
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
            const oCreateModel = new JSONModel(oCreateData);
            this.base.getView().setModel(oCreateModel, "create");
        },

        onCreateCalculationTask() {
            const oView = this.getView();

            if (!this._oCreateDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent10.ext.fragment.CreateCalculationTask",
                    controller: this
                }).then((oDialog) => {
                    this._oCreateDialog = oDialog;
                    oView.addDependent(this._oCreateDialog);
                    this._oCreateDialog.open();
                });
            } else {
                this._oCreateDialog.open();
            }
        },

        onTaskNameChange(oEvent) {
            const sValue = oEvent.getParameter("value");
            const oCreateModel = this.getView().getModel("create");
            const oData = oCreateModel.getData();

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

        onCalculationTypeChange(oEvent) {
            const sValue = oEvent.getParameter("selectedItem").getKey();
            const oCreateModel = this.getView().getModel("create");
            const oData = oCreateModel.getData();

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

        onFormulaCategoryChange(oEvent) {
            const sValue = oEvent.getParameter("selectedItem").getKey();
            const oCreateModel = this.getView().getModel("create");
            const oData = oCreateModel.getData();

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

        onPriorityChange(oEvent) {
            const sValue = oEvent.getParameter("item").getKey();
            const oCreateModel = this.getView().getModel("create");
            const oData = oCreateModel.getData();

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

        onCancelCreateCalculationTask() {
            this._oCreateDialog.close();
            this._resetCreateModel();
        },

        onConfirmCreateCalculationTask() {
            const oCreateModel = this.getView().getModel("create");
            const oData = oCreateModel.getData();

            // Final validation
            if (!this._validateCreateData(oData)) {
                return;
            }

            this._oCreateDialog.setBusy(true);

            // Sanitize data for security
            const oSanitizedData = {
                taskName: SecurityUtils.sanitizeInput(oData.taskName),
                description: SecurityUtils.sanitizeInput(oData.description),
                calculationType: oData.calculationType,
                formulaCategory: oData.formulaCategory,
                priority: oData.priority,
                precisionLevel: oData.precisionLevel,
                calculationEngine: oData.calculationEngine,
                parallelProcessing: !!oData.parallelProcessing,
                threadCount: parseInt(oData.threadCount, 10) || 4,
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
                    const errorMsg = SecurityUtils.escapeHTML(error.message || "Unknown error");
                    MessageBox.error(`${this.getResourceBundle().getText("error.createTaskFailed") }: ${ errorMsg}`);
                }.bind(this)
            });
        },

        _validateCreateData(oData) {
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

        _resetCreateModel() {
            const oCreateModel = this.getView().getModel("create");
            const oData = oCreateModel.getData();

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

        onFormulaChange(oEvent) {
            const sValue = oEvent.getParameter("value");
            const oCreateModel = this.getView().getModel("create");
            const oData = oCreateModel.getData();

            // Basic formula validation
            if (sValue && sValue.length > 0) {
                // Check for basic syntax errors
                try {
                    // Simple validation - actual validation would be more complex
                    if (oData.formulaLanguage === "javascript") {
                        // Check for balanced parentheses and brackets
                        const openParen = (sValue.match(/\(/g) || []).length;
                        const closeParen = (sValue.match(/\)/g) || []).length;
                        const openBracket = (sValue.match(/\[/g) || []).length;
                        const closeBracket = (sValue.match(/\]/g) || []).length;
                        const openBrace = (sValue.match(/\{/g) || []).length;
                        const closeBrace = (sValue.match(/\}/g) || []).length;

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

        onFormulaLanguageChange(oEvent) {
            const sValue = oEvent.getParameter("selectedItem").getKey();
            const oCreateModel = this.getView().getModel("create");
            const oData = oCreateModel.getData();

            // Update code editor type based on language
            const oCodeEditor = this.getView().byId("formulaEditor");
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
        onExecuteCalculation() {
            SecurityUtils.checkCalculationAuth("execute").then((authorized) => {
                if (!authorized) {
                    MessageToast.show(this.getResourceBundle().getText("error.notAuthorized"));
                    return;
                }

                const oContext = this.base.getView().getBindingContext();
                const oData = oContext.getObject();

                if (oData.status === "calculating") {
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
            });
        },

        // Validate Result Action
        onValidateResult() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();

            if (!oData.resultValue) {
                MessageToast.show(this.getResourceBundle().getText("error.noResultToValidate"));
                return;
            }

            this._validateCalculationResult(oContext);
        },

        // Optimize Formula Action
        onOptimizeFormula() {
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
                }).then((oDialog) => {
                    this._formulaOptimizer = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadFormulaOptimizationData(oContext);
                    oDialog.open();
                });
            } else {
                this._loadFormulaOptimizationData(oContext);
                this._formulaOptimizer.open();
            }
        },

        // Analyze Performance Action
        onAnalyzePerformance() {
            const oContext = this.base.getView().getBindingContext();

            if (!this._performanceAnalyzer) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent10.ext.fragment.PerformanceAnalyzer",
                    controller: this
                }).then((oDialog) => {
                    this._performanceAnalyzer = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadPerformanceAnalysis(oContext);
                    oDialog.open();
                });
            } else {
                this._loadPerformanceAnalysis(oContext);
                this._performanceAnalyzer.open();
            }
        },

        // Test Precision Action
        onTestPrecision() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();

            if (!oData.formulaExpression) {
                MessageToast.show(this.getResourceBundle().getText("error.noFormulaToTest"));
                return;
            }

            this._runPrecisionTest(oContext);
        },

        // Run Self-Healing Action
        onRunSelfHealing() {
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
        onExportResults() {
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
                }).then((oDialog) => {
                    this._resultExporter = oDialog;
                    this.getView().addDependent(oDialog);
                    oDialog.open();
                });
            } else {
                this._resultExporter.open();
            }
        },

        // Visualize Data Action
        onVisualizeData() {
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
                }).then((oDialog) => {
                    this._dataVisualizer = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadVisualizationData(oContext);
                    oDialog.open();
                });
            } else {
                this._loadVisualizationData(oContext);
                this._dataVisualizer.open();
            }
        },

        // Real-time monitoring initialization
        onAfterRendering() {
            this._initializeCalculationMonitoring();
        },

        _initializeCalculationMonitoring() {
            const oContext = this.base.getView().getBindingContext();
            if (!oContext) {return;}

            const taskId = oContext.getObject().taskId;

            // Subscribe to calculation updates for this specific task
            if (this._eventSource) {
                this._eventSource.close();
            }

            try {
                this._eventSource = SecurityUtils.createSecureEventSource(`http://localhost:8010/calculations/${taskId}/stream`);

                this._eventSource.addEventListener("calculation-progress", (event) => {
                    this._updateCalculationProgress(event.data);
                });

                this._eventSource.addEventListener("calculation-completed", (event) => {
                    this._handleCalculationCompleted(event.data);
                });

                this._eventSource.addEventListener("calculation-error", (event) => {
                    this._handleCalculationError(event.data);
                });

                this._eventSource.addEventListener("self-healing", (event) => {
                    this._handleSelfHealingUpdate(event.data);
                });

            } catch (error) {
                // // console.warn("Server-Sent Events not available, using polling");
                this._initializePolling(taskId);
            }
        },

        _initializePolling(taskId) {
            this._pollInterval = setInterval(() => {
                this._refreshTaskData();
            }, 2000);
        },

        _executeCalculation(oContext) {
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

        _validateCalculationResult(oContext) {
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

        _runPrecisionTest(oContext) {
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

        _runSelfHealing(oContext) {
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

        _loadFormulaOptimizationData(oContext) {
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

        _loadPerformanceAnalysis(oContext) {
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

        _loadVisualizationData(oContext) {
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

        _updateCalculationProgress(data) {
            // Update progress indicators
            const oProgressIndicator = this.getView().byId("calculationProgress");
            if (oProgressIndicator) {
                oProgressIndicator.setPercentValue(data.progress);
                oProgressIndicator.setDisplayValue(`${data.progress}% - ${data.currentStep}`);
            }
        },

        _handleCalculationCompleted(data) {
            MessageToast.show(this.getResourceBundle().getText("msg.calculationCompleted"));
            this._refreshTaskData();

            // Show performance improvement if available
            if (data.performanceImprovement) {
                MessageToast.show(this.getResourceBundle().getText("msg.performanceImproved", [data.performanceImprovement]));
            }
        },

        _handleCalculationError(data) {
            MessageBox.error(this.getResourceBundle().getText("error.calculationFailed", [SecurityUtils.escapeHTML(data.error)]));
            this._refreshTaskData();
        },

        _handleSelfHealingUpdate(data) {
            MessageToast.show(this.getResourceBundle().getText("msg.selfHealingUpdate", [data.action]));
            this._refreshTaskData();
        },

        _refreshTaskData() {
            const oContext = this.base.getView().getBindingContext();
            if (oContext) {
                oContext.refresh();
            }
        },

        _displayOptimizationSuggestions(data) {
            // Display optimization suggestions in dialog
        },

        _displayPerformanceAnalysis(data) {
            // Display performance analysis charts
        },

        _createDataVisualization(data) {
            // Create data visualization charts
        },

        getResourceBundle() {
            return this.getView().getModel("i18n").getResourceBundle();
        },

        /**
         * @function _startMetricsMonitoring
         * @description Starts real-time metrics monitoring for a specific task.
         * @param {string} sTaskId - Task ID to monitor
         * @param {sap.m.Dialog} oDialog - Metrics dialog
         * @private
         */
        _startMetricsMonitoring(sTaskId, oDialog) {
            // Initialize metrics model
            const oMetricsModel = new JSONModel({
                taskId: sTaskId,
                isMonitoring: true,
                lastUpdated: new Date().toISOString(),
                metrics: {
                    cpu: { current: 0, average: 0, peak: 0 },
                    memory: { current: 0, average: 0, peak: 0 },
                    network: { inbound: 0, outbound: 0 },
                    disk: { read: 0, write: 0 },
                    responseTime: { current: 0, average: 0 },
                    throughput: { current: 0, average: 0 },
                    errorRate: { current: 0, average: 0 }
                },
                alerts: [],
                events: []
            });
            oDialog.setModel(oMetricsModel, "metrics");

            // Start real-time monitoring
            this._initializeMetricsStream(sTaskId, oDialog);
        },

        /**
         * @function _initializeMetricsStream
         * @description Initializes real-time metrics streaming.
         * @param {string} sTaskId - Task ID to monitor
         * @param {sap.m.Dialog} oDialog - Metrics dialog
         * @private
         */
        _initializeMetricsStream(sTaskId, oDialog) {
            if (this._metricsEventSource) {
                this._metricsEventSource.close();
            }

            try {
                this._metricsEventSource = new EventSource(`/api/agent10/monitoring/${sTaskId}/metrics-stream`);

                this._metricsEventSource.onmessage = function(event) {
                    try {
                        const data = JSON.parse(event.data);
                        this._updateMetricsDisplay(data, oDialog);
                    } catch (error) {
                        // // console.error("Error parsing metrics data:", error);
                    }
                }.bind(this);

                this._metricsEventSource.onerror = function(error) {
                    // // console.warn("Metrics stream error, falling back to polling:", error);
                    this._startMetricsPolling(sTaskId, oDialog);
                }.bind(this);

            } catch (error) {
                // // console.warn("EventSource not available, using polling fallback");
                this._startMetricsPolling(sTaskId, oDialog);
            }
        },

        /**
         * @function _startMetricsPolling
         * @description Starts polling fallback for metrics updates.
         * @param {string} sTaskId - Task ID to monitor
         * @param {sap.m.Dialog} oDialog - Metrics dialog
         * @private
         */
        _startMetricsPolling(sTaskId, oDialog) {
            if (this._metricsPollingInterval) {
                clearInterval(this._metricsPollingInterval);
            }

            this._metricsPollingInterval = setInterval(() => {
                this._fetchMetricsData(sTaskId, oDialog);
            }, 2000);
        },

        /**
         * @function _fetchMetricsData
         * @description Fetches metrics data via polling.
         * @param {string} sTaskId - Task ID to monitor
         * @param {sap.m.Dialog} oDialog - Metrics dialog
         * @private
         */
        _fetchMetricsData(sTaskId, oDialog) {
            const oModel = this.base.getView().getModel();

            SecurityUtils.secureCallFunction(oModel, "/GetTaskMetrics", {
                urlParameters: { taskId: sTaskId },
                success: function(data) {
                    this._updateMetricsDisplay(data, oDialog);
                }.bind(this),
                error(error) {
                    // // console.warn("Failed to fetch metrics data:", error);
                }
            });
        },

        /**
         * @function _updateMetricsDisplay
         * @description Updates metrics display with new data.
         * @param {Object} data - Metrics data
         * @param {sap.m.Dialog} oDialog - Metrics dialog
         * @private
         */
        _updateMetricsDisplay(data, oDialog) {
            if (!oDialog || !oDialog.isOpen()) {return;}

            const oMetricsModel = oDialog.getModel("metrics");
            if (oMetricsModel) {
                const oCurrentData = oMetricsModel.getData();
                oCurrentData.metrics = data.metrics || oCurrentData.metrics;
                oCurrentData.lastUpdated = new Date().toISOString();

                if (data.alerts && data.alerts.length > 0) {
                    oCurrentData.alerts = data.alerts;
                }

                if (data.events && data.events.length > 0) {
                    oCurrentData.events = data.events.concat(oCurrentData.events).slice(0, 100); // Keep last 100 events
                }

                oMetricsModel.setData(oCurrentData);
            }
        },

        /**
         * @function _loadAlertSetupData
         * @description Loads alert setup configuration data.
         * @param {string} sTaskId - Task ID
         * @param {sap.m.Dialog} oDialog - Alert setup dialog
         * @private
         */
        _loadAlertSetupData(sTaskId, oDialog) {
            oDialog.setBusy(true);

            const oModel = this.base.getView().getModel();

            SecurityUtils.secureCallFunction(oModel, "/GetTaskAlertConfig", {
                urlParameters: { taskId: sTaskId },
                success: function(data) {
                    const oAlertModel = new JSONModel({
                        taskId: sTaskId,
                        alerts: data.alerts || [],
                        thresholds: data.thresholds || {
                            cpu: { warning: 70, critical: 90 },
                            memory: { warning: 80, critical: 95 },
                            responseTime: { warning: 1000, critical: 5000 },
                            errorRate: { warning: 1, critical: 5 }
                        },
                        notificationChannels: data.channels || [],
                        escalationPolicies: data.escalationPolicies || []
                    });
                    oDialog.setModel(oAlertModel, "alertSetup");
                    oDialog.setBusy(false);
                }.bind(this),
                error(error) {
                    oDialog.setBusy(false);
                    MessageBox.error(`Failed to load alert configuration: ${ error.message}`);
                }
            });
        },

        /**
         * @function _loadMonitoringLogs
         * @description Loads monitoring logs for a specific task.
         * @param {string} sTaskId - Task ID
         * @param {sap.m.Dialog} oDialog - Logs dialog
         * @private
         */
        _loadMonitoringLogs(sTaskId, oDialog) {
            oDialog.setBusy(true);

            const oModel = this.base.getView().getModel();

            SecurityUtils.secureCallFunction(oModel, "/GetTaskLogs", {
                urlParameters: {
                    taskId: sTaskId,
                    limit: 100,
                    level: "INFO"
                },
                success: function(data) {
                    const oLogsModel = oDialog.getModel("logs");
                    if (oLogsModel) {
                        const oCurrentData = oLogsModel.getData();
                        oCurrentData.logs = data.logs || [];
                        oCurrentData.totalCount = data.totalCount || 0;
                        oCurrentData.lastUpdated = new Date().toISOString();
                        oLogsModel.setData(oCurrentData);
                    }
                    oDialog.setBusy(false);

                    // Start auto-refresh if enabled
                    const oLogsData = oDialog.getModel("logs").getData();
                    if (oLogsData.autoRefresh) {
                        this._startLogsAutoRefresh(sTaskId, oDialog);
                    }
                }.bind(this),
                error(error) {
                    oDialog.setBusy(false);
                    MessageBox.error(`Failed to load monitoring logs: ${ error.message}`);
                }
            });
        },

        /**
         * @function _startLogsAutoRefresh
         * @description Starts auto-refresh for monitoring logs.
         * @param {string} sTaskId - Task ID
         * @param {sap.m.Dialog} oDialog - Logs dialog
         * @private
         */
        _startLogsAutoRefresh(sTaskId, oDialog) {
            if (this._logsRefreshInterval) {
                clearInterval(this._logsRefreshInterval);
            }

            const oLogsData = oDialog.getModel("logs").getData();
            this._logsRefreshInterval = setInterval(() => {
                if (oDialog.isOpen()) {
                    this._loadMonitoringLogs(sTaskId, oDialog);
                } else {
                    clearInterval(this._logsRefreshInterval);
                }
            }, oLogsData.refreshInterval || 5000);
        },

        /**
         * @function _getOrCreateDialog
         * @description Gets cached dialog or creates new one with accessibility and responsive features.
         * @param {string} sDialogId - Dialog identifier
         * @param {string} sFragmentName - Fragment name
         * @returns {Promise<sap.m.Dialog>} Promise resolving to dialog
         * @private
         */
        _getOrCreateDialog(sDialogId, sFragmentName) {
            const that = this;

            if (this._dialogCache && this._dialogCache[sDialogId]) {
                return Promise.resolve(this._dialogCache[sDialogId]);
            }

            if (!this._dialogCache) {
                this._dialogCache = {};
            }

            return Fragment.load({
                id: this.base.getView().getId(),
                name: sFragmentName,
                controller: this
            }).then((oDialog) => {
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
        _enableDialogAccessibility(oDialog) {
            oDialog.addEventDelegate({
                onAfterRendering() {
                    const $dialog = oDialog.$();

                    // Set tabindex for focusable elements
                    $dialog.find("input, button, select, textarea").attr("tabindex", "0");

                    // Handle escape key
                    $dialog.on("keydown", (e) => {
                        if (e.key === "Escape") {
                            oDialog.close();
                        }
                    });

                    // Focus first input on open
                    setTimeout(() => {
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
        _optimizeDialogForDevice(oDialog) {
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
        _initializeSecurity() {
            this._auditLogger = {
                log: function(action, details) {
                    const user = this._getCurrentUser();
                    const timestamp = new Date().toISOString();
                    const _logEntry = {
                        timestamp,
                        user,
                        agent: "Agent10_Monitoring",
                        action,
                        details: details || {}
                    };
                    // // console.info(`AUDIT: ${ JSON.stringify(_logEntry)}`);
                }.bind(this)
            };
        },

        /**
         * @function _getCurrentUser
         * @description Gets current user ID for audit logging.
         * @returns {string} User ID or "anonymous"
         * @private
         */
        _getCurrentUser() {
            return sap.ushell?.Container?.getUser()?.getId() || "anonymous";
        },

        /**
         * @function _hasRole
         * @description Checks if current user has specified role.
         * @param {string} role - Role to check
         * @returns {boolean} True if user has role
         * @private
         */
        _hasRole(role) {
            const user = sap.ushell?.Container?.getUser();
            if (user && user.hasRole) {
                return user.hasRole(role);
            }
            // Mock role validation for development/testing
            const mockRoles = ["MonitoringAdmin", "MonitoringUser", "MonitoringOperator"];
            return mockRoles.includes(role);
        },

        /**
         * @function _cleanupResources
         * @description Cleans up resources to prevent memory leaks.
         * @private
         */
        _cleanupResources() {
            // Clean up EventSource connections
            if (this._metricsEventSource) {
                this._metricsEventSource.close();
                this._metricsEventSource = null;
            }

            // Clean up polling intervals
            if (this._metricsPollingInterval) {
                clearInterval(this._metricsPollingInterval);
                this._metricsPollingInterval = null;
            }

            if (this._logsRefreshInterval) {
                clearInterval(this._logsRefreshInterval);
                this._logsRefreshInterval = null;
            }

            // Clean up EventSource from original implementation
            if (this._eventSource) {
                this._eventSource.close();
                this._eventSource = null;
            }

            if (this._pollInterval) {
                clearInterval(this._pollInterval);
                this._pollInterval = null;
            }

            // Clean up cached dialogs
            if (this._dialogCache) {
                Object.keys(this._dialogCache).forEach((key) => {
                    if (this._dialogCache[key]) {
                        this._dialogCache[key].destroy();
                    }
                });
                this._dialogCache = {};
            }
        },

        onExit() {
            this._cleanupResources();
        }
    });
});