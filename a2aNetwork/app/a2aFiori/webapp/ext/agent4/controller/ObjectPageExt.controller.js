sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel"
], function (ControllerExtension, MessageBox, MessageToast, Fragment, JSONModel) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent4.ext.controller.ObjectPageExt", {
        
        override: {
            onInit: function () {
                this._extensionAPI = this.base.getExtensionAPI();
            }
        },

        onStartValidation: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            var sTaskName = oContext.getProperty("taskName");
            
            MessageBox.confirm("Start calculation validation for '" + sTaskName + "'?", {
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        this._startValidationProcess(sTaskId);
                    }
                }.bind(this)
            });
        },

        _startValidationProcess: function(sTaskId) {
            this._extensionAPI.getView().setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent4/v1/tasks/" + sTaskId + "/validate",
                type: "POST",
                success: function(data) {
                    this._extensionAPI.getView().setBusy(false);
                    MessageToast.show("Validation process started");
                    this._extensionAPI.refresh();
                    
                    // Start progress monitoring
                    this._startValidationMonitoring(sTaskId);
                }.bind(this),
                error: function(xhr) {
                    this._extensionAPI.getView().setBusy(false);
                    MessageBox.error("Failed to start validation: " + xhr.responseText);
                }.bind(this)
            });
        },

        _startValidationMonitoring: function(sTaskId) {
            // Poll for validation progress
            this._validationInterval = setInterval(function() {
                jQuery.ajax({
                    url: "/a2a/agent4/v1/tasks/" + sTaskId + "/progress",
                    type: "GET",
                    success: function(data) {
                        if (data.status === "COMPLETED" || data.status === "FAILED") {
                            clearInterval(this._validationInterval);
                            this._extensionAPI.refresh();
                            
                            if (data.status === "COMPLETED") {
                                MessageBox.success(
                                    "Validation completed!\n" +
                                    "Formulas validated: " + data.formulasValidated + "\n" +
                                    "Accuracy: " + data.accuracy + "%\n" +
                                    "Errors found: " + data.errorCount
                                );
                            } else {
                                MessageBox.error("Validation failed: " + data.error);
                            }
                        } else {
                            // Update progress indicator
                            MessageToast.show("Validating: " + data.progress + "%");
                        }
                    }.bind(this),
                    error: function() {
                        clearInterval(this._validationInterval);
                    }.bind(this)
                });
            }.bind(this), 2000);
        },

        onValidateFormulas: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            
            if (!this._oFormulaValidationDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent4.ext.fragment.FormulaValidation",
                    controller: this
                }).then(function(oDialog) {
                    this._oFormulaValidationDialog = oDialog;
                    this.base.getView().addDependent(this._oFormulaValidationDialog);
                    
                    // Load formulas for this task
                    this._loadTaskFormulas(sTaskId);
                    this._oFormulaValidationDialog.open();
                }.bind(this));
            } else {
                this._loadTaskFormulas(sTaskId);
                this._oFormulaValidationDialog.open();
            }
        },

        _loadTaskFormulas: function(sTaskId) {
            this._oFormulaValidationDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent4/v1/tasks/" + sTaskId + "/formulas",
                type: "GET",
                success: function(data) {
                    this._oFormulaValidationDialog.setBusy(false);
                    
                    var oModel = new JSONModel({
                        taskId: sTaskId,
                        formulas: data.formulas,
                        validationSettings: {
                            precisionThreshold: 0.001,
                            toleranceLevel: 0.01,
                            testCases: 10
                        }
                    });
                    this._oFormulaValidationDialog.setModel(oModel, "validation");
                }.bind(this),
                error: function(xhr) {
                    this._oFormulaValidationDialog.setBusy(false);
                    MessageBox.error("Failed to load formulas: " + xhr.responseText);
                }.bind(this)
            });
        },

        onValidateSelectedFormulas: function() {
            var oModel = this._oFormulaValidationDialog.getModel("validation");
            var aFormulas = oModel.getProperty("/formulas");
            var aSelectedFormulas = aFormulas.filter(function(formula) {
                return formula.selected;
            });
            
            if (aSelectedFormulas.length === 0) {
                MessageBox.error("Please select formulas to validate");
                return;
            }
            
            this._oFormulaValidationDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent4/v1/validate-formulas",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    formulas: aSelectedFormulas,
                    settings: oModel.getProperty("/validationSettings")
                }),
                success: function(data) {
                    this._oFormulaValidationDialog.setBusy(false);
                    
                    // Update formula results
                    this._updateFormulaResults(data.results);
                    
                    MessageBox.success(
                        "Formula validation completed!\n" +
                        "Validated: " + data.validated + "/" + aSelectedFormulas.length + "\n" +
                        "Passed: " + data.passed + "\n" +
                        "Failed: " + data.failed
                    );
                }.bind(this),
                error: function(xhr) {
                    this._oFormulaValidationDialog.setBusy(false);
                    MessageBox.error("Formula validation failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        _updateFormulaResults: function(aResults) {
            var oModel = this._oFormulaValidationDialog.getModel("validation");
            var aFormulas = oModel.getProperty("/formulas");
            
            aResults.forEach(function(result) {
                var oFormula = aFormulas.find(function(f) {
                    return f.id === result.formulaId;
                });
                if (oFormula) {
                    oFormula.validationResult = result.result;
                    oFormula.actualValue = result.actualValue;
                    oFormula.variance = result.variance;
                    oFormula.errors = result.errors;
                }
            });
            
            oModel.setProperty("/formulas", aFormulas);
        },

        onRunBenchmark: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            
            MessageBox.confirm(
                "Run performance benchmark? This may take several minutes.",
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._runPerformanceBenchmark(sTaskId);
                        }
                    }.bind(this)
                }
            );
        },

        _runPerformanceBenchmark: function(sTaskId) {
            this._extensionAPI.getView().setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent4/v1/tasks/" + sTaskId + "/benchmark",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    iterations: 1000,
                    measureMemory: true,
                    measureCPU: true,
                    compareWith: "reference"
                }),
                success: function(data) {
                    this._extensionAPI.getView().setBusy(false);
                    this._showBenchmarkResults(data);
                }.bind(this),
                error: function(xhr) {
                    this._extensionAPI.getView().setBusy(false);
                    MessageBox.error("Benchmark failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        _showBenchmarkResults: function(data) {
            var sMessage = "Benchmark Results:\n\n" +
                          "Average execution time: " + data.avgExecutionTime + " ms\n" +
                          "Memory usage: " + data.memoryUsage + " MB\n" +
                          "CPU efficiency: " + data.cpuEfficiency + "%\n" +
                          "Formulas per second: " + data.formulasPerSecond;
            
            MessageBox.information(sMessage, {
                title: "Performance Benchmark"
            });
        },

        onGenerateReport: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            var sTaskName = oContext.getProperty("taskName");
            
            if (!this._oReportDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent4.ext.fragment.ValidationReport",
                    controller: this
                }).then(function(oDialog) {
                    this._oReportDialog = oDialog;
                    this.base.getView().addDependent(this._oReportDialog);
                    
                    var oModel = new JSONModel({
                        taskId: sTaskId,
                        taskName: sTaskName,
                        reportType: "DETAILED",
                        includeFormulas: true,
                        includeErrors: true,
                        includePerformance: true,
                        format: "PDF"
                    });
                    this._oReportDialog.setModel(oModel, "report");
                    this._oReportDialog.open();
                }.bind(this));
            } else {
                this._oReportDialog.open();
            }
        },

        onGenerateValidationReport: function() {
            var oModel = this._oReportDialog.getModel("report");
            var oData = oModel.getData();
            
            this._oReportDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent4/v1/tasks/" + oData.taskId + "/report",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    type: oData.reportType,
                    includeFormulas: oData.includeFormulas,
                    includeErrors: oData.includeErrors,
                    includePerformance: oData.includePerformance,
                    format: oData.format
                }),
                success: function(data) {
                    this._oReportDialog.setBusy(false);
                    this._oReportDialog.close();
                    
                    MessageBox.success(
                        "Validation report generated successfully!",
                        {
                            actions: ["Download", MessageBox.Action.CLOSE],
                            onClose: function(oAction) {
                                if (oAction === "Download") {
                                    window.open(data.downloadUrl, "_blank");
                                }
                            }
                        }
                    );
                }.bind(this),
                error: function(xhr) {
                    this._oReportDialog.setBusy(false);
                    MessageBox.error("Report generation failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        onOptimizeCalculations: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            
            MessageBox.confirm(
                "Optimize calculation performance? This will analyze and suggest improvements.",
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._optimizeCalculations(sTaskId);
                        }
                    }.bind(this)
                }
            );
        },

        _optimizeCalculations: function(sTaskId) {
            this._extensionAPI.getView().setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent4/v1/tasks/" + sTaskId + "/optimize",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    optimizeFor: "SPEED",
                    preserveAccuracy: true,
                    maxIterations: 100
                }),
                success: function(data) {
                    this._extensionAPI.getView().setBusy(false);
                    
                    if (data.optimizations.length > 0) {
                        this._showOptimizationSuggestions(data);
                    } else {
                        MessageBox.information("No optimization opportunities found. Calculations are already optimized.");
                    }
                }.bind(this),
                error: function(xhr) {
                    this._extensionAPI.getView().setBusy(false);
                    MessageBox.error("Optimization failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        _showOptimizationSuggestions: function(data) {
            var oView = this.base.getView();
            
            if (!this._oOptimizationDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent4.ext.fragment.OptimizationSuggestions",
                    controller: this
                }).then(function(oDialog) {
                    this._oOptimizationDialog = oDialog;
                    oView.addDependent(this._oOptimizationDialog);
                    
                    var oModel = new JSONModel(data);
                    this._oOptimizationDialog.setModel(oModel, "optimization");
                    this._oOptimizationDialog.open();
                }.bind(this));
            } else {
                var oModel = new JSONModel(data);
                this._oOptimizationDialog.setModel(oModel, "optimization");
                this._oOptimizationDialog.open();
            }
        },

        onApplyOptimization: function(oEvent) {
            var oSource = oEvent.getSource();
            var oBindingContext = oSource.getBindingContext("optimization");
            var oOptimization = oBindingContext.getObject();
            
            MessageBox.confirm(
                "Apply optimization: " + oOptimization.description + "?\n" +
                "Expected improvement: " + oOptimization.expectedImprovement,
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._applyOptimization(oOptimization);
                        }
                    }.bind(this)
                }
            );
        },

        _applyOptimization: function(oOptimization) {
            jQuery.ajax({
                url: "/a2a/agent4/v1/apply-optimization",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify(oOptimization),
                success: function(data) {
                    MessageToast.show("Optimization applied successfully");
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to apply optimization: " + xhr.responseText);
                }
            });
        }
    });
});