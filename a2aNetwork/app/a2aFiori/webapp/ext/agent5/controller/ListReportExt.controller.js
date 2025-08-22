sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/ui/core/Fragment",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/model/json/JSONModel"
], function (ControllerExtension, Fragment, MessageBox, MessageToast, JSONModel) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent5.ext.controller.ListReportExt", {
        
        override: {
            onInit: function () {
                this._extensionAPI = this.base.getExtensionAPI();
            }
        },

        onCreateQATask: function() {
            var oView = this.base.getView();
            
            if (!this._oCreateDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent5.ext.fragment.CreateQATask",
                    controller: this
                }).then(function(oDialog) {
                    this._oCreateDialog = oDialog;
                    oView.addDependent(this._oCreateDialog);
                    
                    // Initialize model
                    var oModel = new JSONModel({
                        taskName: "",
                        description: "",
                        testSuite: "",
                        testType: "UNIT",
                        targetApplication: "",
                        testFramework: "SELENIUM",
                        testEnvironment: "TEST",
                        severity: "MEDIUM",
                        automationLevel: 80,
                        parallelExecution: true,
                        retryOnFailure: true,
                        complianceStandard: "",
                        testCases: []
                    });
                    this._oCreateDialog.setModel(oModel, "create");
                    this._oCreateDialog.open();
                }.bind(this));
            } else {
                this._oCreateDialog.open();
            }
        },

        onTestRunner: function() {
            var oView = this.base.getView();
            
            if (!this._oTestRunnerDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent5.ext.fragment.TestRunner",
                    controller: this
                }).then(function(oDialog) {
                    this._oTestRunnerDialog = oDialog;
                    oView.addDependent(this._oTestRunnerDialog);
                    
                    // Initialize test runner model
                    var oModel = new JSONModel({
                        selectedSuite: "",
                        executionMode: "SEQUENTIAL",
                        environment: "TEST",
                        parallelThreads: 4,
                        retryAttempts: 3,
                        generateReport: true,
                        notifyOnCompletion: true,
                        testFilters: {
                            includeSmoke: true,
                            includeRegression: false,
                            priorityFilter: "ALL"
                        },
                        executionResults: []
                    });
                    this._oTestRunnerDialog.setModel(oModel, "runner");
                    this._oTestRunnerDialog.open();
                    
                    // Load available test suites
                    this._loadTestSuites();
                }.bind(this));
            } else {
                this._oTestRunnerDialog.open();
                this._loadTestSuites();
            }
        },

        _loadTestSuites: function() {
            jQuery.ajax({
                url: "/a2a/agent5/v1/test-suites",
                type: "GET",
                success: function(data) {
                    var oRunnerModel = this._oTestRunnerDialog.getModel("runner");
                    oRunnerModel.setProperty("/availableSuites", data.suites);
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to load test suites");
                }
            });
        },

        onExecuteTestSuite: function() {
            var oModel = this._oTestRunnerDialog.getModel("runner");
            var oData = oModel.getData();
            
            if (!oData.selectedSuite) {
                MessageBox.error("Please select a test suite to execute");
                return;
            }
            
            this._oTestRunnerDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent5/v1/execute-test-suite",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    suiteId: oData.selectedSuite,
                    executionMode: oData.executionMode,
                    environment: oData.environment,
                    parallelThreads: oData.parallelThreads,
                    retryAttempts: oData.retryAttempts,
                    filters: oData.testFilters
                }),
                success: function(data) {
                    this._oTestRunnerDialog.setBusy(false);
                    
                    MessageBox.success(
                        "Test execution started!\n" +
                        "Execution ID: " + data.executionId + "\n" +
                        "Estimated time: " + data.estimatedTime + " minutes"
                    );
                    
                    // Start monitoring execution
                    this._monitorTestExecution(data.executionId);
                }.bind(this),
                error: function(xhr) {
                    this._oTestRunnerDialog.setBusy(false);
                    MessageBox.error("Test execution failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        _monitorTestExecution: function(executionId) {
            // Poll for execution status updates
            var iInterval = setInterval(function() {
                jQuery.ajax({
                    url: "/a2a/agent5/v1/execution-status/" + executionId,
                    type: "GET",
                    success: function(data) {
                        var oModel = this._oTestRunnerDialog.getModel("runner");
                        oModel.setProperty("/executionResults", data.results);
                        
                        if (data.status === "COMPLETED" || data.status === "FAILED") {
                            clearInterval(iInterval);
                            
                            if (data.status === "COMPLETED") {
                                MessageBox.success(
                                    "Test execution completed!\n" +
                                    "Passed: " + data.passed + "\n" +
                                    "Failed: " + data.failed + "\n" +
                                    "Success Rate: " + data.successRate + "%"
                                );
                            } else {
                                MessageBox.error("Test execution failed: " + data.error);
                            }
                            
                            this._extensionAPI.refresh();
                        }
                    }.bind(this),
                    error: function() {
                        clearInterval(iInterval);
                        MessageBox.error("Lost connection to test execution");
                    }
                });
            }.bind(this), 5000);
        },

        onQualityDashboard: function() {
            // Navigate to quality dashboard
            var oRouter = sap.ui.core.UIComponent.getRouterFor(this.base.getView());
            oRouter.navTo("QualityDashboard");
        },

        onBatchExecution: function() {
            var oTable = this._extensionAPI.getTable();
            var aSelectedContexts = oTable.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageBox.warning("Please select tasks for batch execution.");
                return;
            }
            
            MessageBox.confirm(
                "Execute tests for " + aSelectedContexts.length + " tasks?",
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._startBatchExecution(aSelectedContexts);
                        }
                    }.bind(this)
                }
            );
        },

        _startBatchExecution: function(aContexts) {
            var aTaskIds = aContexts.map(function(oContext) {
                return oContext.getProperty("ID");
            });
            
            this.base.getView().setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent5/v1/batch-execute",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    taskIds: aTaskIds,
                    executionMode: "PARALLEL",
                    priority: "HIGH",
                    generateConsolidatedReport: true
                }),
                success: function(data) {
                    this.base.getView().setBusy(false);
                    MessageBox.success(
                        "Batch execution started!\n" +
                        "Job ID: " + data.jobId + "\n" +
                        "Total test cases: " + data.totalTestCases
                    );
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    this.base.getView().setBusy(false);
                    MessageBox.error("Batch execution failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        onDefectTracking: function() {
            var oView = this.base.getView();
            
            if (!this._oDefectTrackingDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent5.ext.fragment.DefectTracking",
                    controller: this
                }).then(function(oDialog) {
                    this._oDefectTrackingDialog = oDialog;
                    oView.addDependent(this._oDefectTrackingDialog);
                    this._oDefectTrackingDialog.open();
                    
                    // Load defect statistics
                    this._loadDefectStatistics();
                }.bind(this));
            } else {
                this._oDefectTrackingDialog.open();
                this._loadDefectStatistics();
            }
        },

        _loadDefectStatistics: function() {
            this._oDefectTrackingDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent5/v1/defect-statistics",
                type: "GET",
                success: function(data) {
                    this._oDefectTrackingDialog.setBusy(false);
                    
                    var oModel = new JSONModel(data);
                    this._oDefectTrackingDialog.setModel(oModel, "defects");
                    
                    // Create defect charts
                    this._createDefectCharts(data);
                }.bind(this),
                error: function(xhr) {
                    this._oDefectTrackingDialog.setBusy(false);
                    MessageBox.error("Failed to load defect statistics");
                }.bind(this)
            });
        },

        _createDefectCharts: function(data) {
            // Create defect trend charts, severity distribution, etc.
            // Using SAP Viz framework
        },

        onConfirmCreateTask: function() {
            var oModel = this._oCreateDialog.getModel("create");
            var oData = oModel.getData();
            
            // Validation
            if (!oData.taskName || !oData.testType || !oData.targetApplication) {
                MessageBox.error("Please fill all required fields");
                return;
            }
            
            if (oData.testCases.length === 0) {
                MessageBox.warning("No test cases defined. Task will be created but cannot be executed until test cases are added.");
            }
            
            this._oCreateDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent5/v1/tasks",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify(oData),
                success: function(data) {
                    this._oCreateDialog.setBusy(false);
                    this._oCreateDialog.close();
                    MessageToast.show("QA validation task created successfully");
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

        onAddTestCase: function() {
            var oModel = this._oCreateDialog.getModel("create");
            var aTestCases = oModel.getProperty("/testCases");
            
            aTestCases.push({
                testName: "",
                description: "",
                category: "FUNCTIONAL",
                priority: "MEDIUM",
                automationLevel: "MANUAL",
                expectedResult: "",
                testSteps: []
            });
            
            oModel.setProperty("/testCases", aTestCases);
        },

        onImportTestCases: function() {
            // Show file upload dialog for importing test cases
            MessageBox.information("Test case import functionality will be available soon.");
        }
    });
});