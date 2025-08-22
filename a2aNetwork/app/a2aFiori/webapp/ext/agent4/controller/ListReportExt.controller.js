sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/ui/core/Fragment",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/model/json/JSONModel"
], function (ControllerExtension, Fragment, MessageBox, MessageToast, JSONModel) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent4.ext.controller.ListReportExt", {
        
        override: {
            onInit: function () {
                this._extensionAPI = this.base.getExtensionAPI();
            }
        },

        onCreateValidationTask: function() {
            var oView = this.base.getView();
            
            if (!this._oCreateDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent4.ext.fragment.CreateValidationTask",
                    controller: this
                }).then(function(oDialog) {
                    this._oCreateDialog = oDialog;
                    oView.addDependent(this._oCreateDialog);
                    
                    // Initialize model
                    var oModel = new JSONModel({
                        taskName: "",
                        description: "",
                        calculationType: "MATHEMATICAL",
                        dataSource: "",
                        priority: "MEDIUM",
                        validationMode: "STANDARD",
                        precisionThreshold: 0.001,
                        toleranceLevel: 0.01,
                        enableCrossValidation: true,
                        enableStatisticalTests: false,
                        formulas: []
                    });
                    this._oCreateDialog.setModel(oModel, "create");
                    this._oCreateDialog.open();
                }.bind(this));
            } else {
                this._oCreateDialog.open();
            }
        },

        onFormulaBuilder: function() {
            var oView = this.base.getView();
            
            if (!this._oFormulaBuilderDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent4.ext.fragment.FormulaBuilder",
                    controller: this
                }).then(function(oDialog) {
                    this._oFormulaBuilderDialog = oDialog;
                    oView.addDependent(this._oFormulaBuilderDialog);
                    
                    // Initialize formula builder model
                    var oModel = new JSONModel({
                        currentFormula: "",
                        variables: [],
                        functions: [
                            "SUM", "AVERAGE", "MIN", "MAX", "COUNT",
                            "SQRT", "POW", "LOG", "EXP", "ABS",
                            "SIN", "COS", "TAN", "PI", "E"
                        ],
                        operators: ["+", "-", "*", "/", "^", "(", ")"],
                        testResults: [],
                        syntaxValid: false
                    });
                    this._oFormulaBuilderDialog.setModel(oModel, "formula");
                    this._oFormulaBuilderDialog.open();
                }.bind(this));
            } else {
                this._oFormulaBuilderDialog.open();
            }
        },

        onBatchValidation: function() {
            var oTable = this._extensionAPI.getTable();
            var aSelectedContexts = oTable.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageBox.warning("Please select tasks for batch validation.");
                return;
            }
            
            MessageBox.confirm(
                "Start batch validation for " + aSelectedContexts.length + " tasks?",
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._startBatchValidation(aSelectedContexts);
                        }
                    }.bind(this)
                }
            );
        },

        _startBatchValidation: function(aContexts) {
            var aTaskIds = aContexts.map(function(oContext) {
                return oContext.getProperty("ID");
            });
            
            this.base.getView().setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent4/v1/batch-validate",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    taskIds: aTaskIds,
                    parallel: true,
                    priority: "HIGH",
                    validationSettings: {
                        precisionThreshold: 0.001,
                        enableCrossValidation: true
                    }
                }),
                success: function(data) {
                    this.base.getView().setBusy(false);
                    MessageBox.success(
                        "Batch validation started!\n" +
                        "Job ID: " + data.jobId + "\n" +
                        "Estimated formulas: " + data.totalFormulas
                    );
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    this.base.getView().setBusy(false);
                    MessageBox.error("Batch validation failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        onCalculationTemplates: function() {
            // Navigate to calculation templates
            var oRouter = sap.ui.core.UIComponent.getRouterFor(this.base.getView());
            oRouter.navTo("CalculationTemplates");
        },

        onValidationAnalytics: function() {
            var oView = this.base.getView();
            
            if (!this._oAnalyticsDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent4.ext.fragment.ValidationAnalytics",
                    controller: this
                }).then(function(oDialog) {
                    this._oAnalyticsDialog = oDialog;
                    oView.addDependent(this._oAnalyticsDialog);
                    this._oAnalyticsDialog.open();
                    
                    // Load analytics data
                    this._loadValidationAnalytics();
                }.bind(this));
            } else {
                this._oAnalyticsDialog.open();
                this._loadValidationAnalytics();
            }
        },

        _loadValidationAnalytics: function() {
            this._oAnalyticsDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent4/v1/analytics",
                type: "GET",
                success: function(data) {
                    this._oAnalyticsDialog.setBusy(false);
                    
                    var oModel = new JSONModel(data);
                    this._oAnalyticsDialog.setModel(oModel, "analytics");
                    
                    // Create charts
                    this._createAnalyticsCharts(data);
                }.bind(this),
                error: function(xhr) {
                    this._oAnalyticsDialog.setBusy(false);
                    MessageBox.error("Failed to load analytics: " + xhr.responseText);
                }.bind(this)
            });
        },

        _createAnalyticsCharts: function(data) {
            // Create accuracy trend chart, error distribution, etc.
            // Using SAP Viz framework
        },

        onConfirmCreateTask: function() {
            var oModel = this._oCreateDialog.getModel("create");
            var oData = oModel.getData();
            
            // Validation
            if (!oData.taskName || !oData.calculationType) {
                MessageBox.error("Please fill all required fields");
                return;
            }
            
            if (oData.formulas.length === 0) {
                MessageBox.error("Please add at least one formula to validate");
                return;
            }
            
            this._oCreateDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent4/v1/tasks",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify(oData),
                success: function(data) {
                    this._oCreateDialog.setBusy(false);
                    this._oCreateDialog.close();
                    MessageToast.show("Validation task created successfully");
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    this._oCreateDialog.setBusy(false);
                    MessageBox.error("Failed to create task: " + xhr.responseText);
                }.bind(this)
            });
        },

        onCancelCreateTask: function() {
            this._oCreateDialog.close();
        },

        // Formula Builder Methods
        onAddFunction: function(oEvent) {
            var sFunction = oEvent.getSource().getText();
            this._insertIntoFormula(sFunction + "()");
        },

        onAddOperator: function(oEvent) {
            var sOperator = oEvent.getSource().getText();
            this._insertIntoFormula(sOperator);
        },

        _insertIntoFormula: function(sText) {
            var oModel = this._oFormulaBuilderDialog.getModel("formula");
            var sCurrentFormula = oModel.getProperty("/currentFormula");
            oModel.setProperty("/currentFormula", sCurrentFormula + sText);
            
            // Validate syntax
            this._validateFormulaSyntax();
        },

        _validateFormulaSyntax: function() {
            var oModel = this._oFormulaBuilderDialog.getModel("formula");
            var sFormula = oModel.getProperty("/currentFormula");
            
            if (!sFormula) {
                oModel.setProperty("/syntaxValid", false);
                return;
            }
            
            jQuery.ajax({
                url: "/a2a/agent4/v1/validate-syntax",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({ formula: sFormula }),
                success: function(data) {
                    oModel.setProperty("/syntaxValid", data.valid);
                    if (!data.valid) {
                        oModel.setProperty("/syntaxError", data.error);
                    }
                }.bind(this),
                error: function() {
                    oModel.setProperty("/syntaxValid", false);
                }
            });
        },

        onTestFormula: function() {
            var oModel = this._oFormulaBuilderDialog.getModel("formula");
            var sFormula = oModel.getProperty("/currentFormula");
            
            if (!sFormula) {
                MessageBox.error("Please enter a formula to test");
                return;
            }
            
            // Show test data input dialog
            this._showFormulaTestDialog(sFormula);
        },

        _showFormulaTestDialog: function(sFormula) {
            var oView = this.base.getView();
            
            if (!this._oTestDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent4.ext.fragment.FormulaTest",
                    controller: this
                }).then(function(oDialog) {
                    this._oTestDialog = oDialog;
                    oView.addDependent(this._oTestDialog);
                    
                    var oModel = new JSONModel({
                        formula: sFormula,
                        testData: "{}",
                        expectedResult: "",
                        actualResult: "",
                        testPassed: false
                    });
                    this._oTestDialog.setModel(oModel, "test");
                    this._oTestDialog.open();
                }.bind(this));
            } else {
                var oModel = this._oTestDialog.getModel("test");
                oModel.setProperty("/formula", sFormula);
                this._oTestDialog.open();
            }
        },

        onExecuteFormulaTest: function() {
            var oModel = this._oTestDialog.getModel("test");
            var oData = oModel.getData();
            
            this._oTestDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent4/v1/test-formula",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    formula: oData.formula,
                    testData: JSON.parse(oData.testData),
                    expectedResult: parseFloat(oData.expectedResult)
                }),
                success: function(data) {
                    this._oTestDialog.setBusy(false);
                    
                    oModel.setProperty("/actualResult", data.result);
                    oModel.setProperty("/testPassed", data.passed);
                    oModel.setProperty("/variance", data.variance);
                    
                    if (data.passed) {
                        MessageToast.show("Formula test passed!");
                    } else {
                        MessageBox.warning("Formula test failed. Check variance: " + data.variance);
                    }
                }.bind(this),
                error: function(xhr) {
                    this._oTestDialog.setBusy(false);
                    MessageBox.error("Formula test failed: " + xhr.responseText);
                }.bind(this)
            });
        }
    });
});