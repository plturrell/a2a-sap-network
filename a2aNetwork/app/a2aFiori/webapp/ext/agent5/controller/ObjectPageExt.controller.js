sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel"
], function (ControllerExtension, MessageBox, MessageToast, Fragment, JSONModel) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent5.ext.controller.ObjectPageExt", {
        
        override: {
            onInit: function () {
                this._extensionAPI = this.base.getExtensionAPI();
            }
        },

        onExecuteTests: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            var sTaskName = oContext.getProperty("taskName");
            var iTotalTests = oContext.getProperty("totalTests");
            
            if (iTotalTests === 0) {
                MessageBox.error("No test cases defined for this task. Please add test cases before execution.");
                return;
            }
            
            MessageBox.confirm("Execute " + iTotalTests + " tests for '" + sTaskName + "'?", {
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        this._executeTestTask(sTaskId);
                    }
                }.bind(this)
            });
        },

        _executeTestTask: function(sTaskId) {
            this._extensionAPI.getView().setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent5/v1/tasks/" + sTaskId + "/execute",
                type: "POST",
                success: function(data) {
                    this._extensionAPI.getView().setBusy(false);
                    MessageToast.show("Test execution started");
                    this._extensionAPI.refresh();
                    
                    // Start real-time monitoring
                    this._startTestExecutionMonitoring(sTaskId, data.executionId);
                }.bind(this),
                error: function(xhr) {
                    this._extensionAPI.getView().setBusy(false);
                    MessageBox.error("Failed to start test execution: " + xhr.responseText);
                }.bind(this)
            });
        },

        _startTestExecutionMonitoring: function(sTaskId, sExecutionId) {
            // WebSocket for real-time test execution updates
            this._ws = new WebSocket("wss://" + window.location.host + "/a2a/agent5/v1/tasks/" + sTaskId + "/execution/" + sExecutionId + "/ws");
            
            this._ws.onmessage = function(event) {
                var data = JSON.parse(event.data);
                
                switch(data.type) {
                    case "test_started":
                        MessageToast.show("Test started: " + data.testName);
                        break;
                    case "test_completed":
                        var sStatus = data.result === "PASS" ? "✓" : "✗";
                        MessageToast.show(sStatus + " " + data.testName + " (" + data.duration + "ms)");
                        break;
                    case "suite_completed":
                        this._ws.close();
                        this._extensionAPI.refresh();
                        this._showExecutionSummary(data);
                        break;
                    case "error":
                        this._ws.close();
                        MessageBox.error("Test execution error: " + data.error);
                        break;
                }
            }.bind(this);
            
            this._ws.onerror = function() {
                MessageBox.error("Lost connection to test execution");
            };
        },

        _showExecutionSummary: function(data) {
            var sMessage = "Test Execution Summary:\n\n" +
                          "Total Tests: " + data.totalTests + "\n" +
                          "Passed: " + data.passedTests + "\n" +
                          "Failed: " + data.failedTests + "\n" +
                          "Skipped: " + data.skippedTests + "\n" +
                          "Success Rate: " + data.successRate + "%\n" +
                          "Execution Time: " + data.totalDuration + "ms";
            
            MessageBox.success(sMessage, {
                title: "Execution Completed"
            });
        },

        onValidateCompliance: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            var sComplianceStandard = oContext.getProperty("complianceStandard");
            
            if (!sComplianceStandard) {
                MessageBox.error("No compliance standard specified for this task");
                return;
            }
            
            if (!this._oComplianceDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent5.ext.fragment.ComplianceValidation",
                    controller: this
                }).then(function(oDialog) {
                    this._oComplianceDialog = oDialog;
                    this.base.getView().addDependent(this._oComplianceDialog);
                    
                    var oModel = new JSONModel({
                        taskId: sTaskId,
                        standard: sComplianceStandard,
                        validationScope: "FULL",
                        generateCertificate: true,
                        includeEvidence: true
                    });
                    this._oComplianceDialog.setModel(oModel, "compliance");
                    this._oComplianceDialog.open();
                }.bind(this));
            } else {
                this._oComplianceDialog.open();
            }
        },

        onExecuteComplianceValidation: function() {
            var oModel = this._oComplianceDialog.getModel("compliance");
            var oData = oModel.getData();
            
            this._oComplianceDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent5/v1/tasks/" + oData.taskId + "/validate-compliance",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    standard: oData.standard,
                    scope: oData.validationScope,
                    generateCertificate: oData.generateCertificate,
                    includeEvidence: oData.includeEvidence
                }),
                success: function(data) {
                    this._oComplianceDialog.setBusy(false);
                    
                    if (data.compliant) {
                        MessageBox.success(
                            "Compliance validation passed!\n" +
                            "Standard: " + data.standard + "\n" +
                            "Score: " + data.complianceScore + "%\n" +
                            "Certificate ID: " + data.certificateId
                        );
                    } else {
                        this._showComplianceIssues(data.issues);
                    }
                    
                    this._oComplianceDialog.close();
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    this._oComplianceDialog.setBusy(false);
                    MessageBox.error("Compliance validation failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        _showComplianceIssues: function(aIssues) {
            var sMessage = "Compliance Issues Found:\n\n";
            aIssues.forEach(function(issue, index) {
                sMessage += (index + 1) + ". " + issue.description + "\n";
                sMessage += "   Severity: " + issue.severity + "\n";
                sMessage += "   Recommendation: " + issue.recommendation + "\n\n";
            });
            
            MessageBox.warning(sMessage, {
                title: "Compliance Validation Failed"
            });
        },

        onGenerateTestReport: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            var sTaskName = oContext.getProperty("taskName");
            
            if (!this._oReportDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent5.ext.fragment.TestReport",
                    controller: this
                }).then(function(oDialog) {
                    this._oReportDialog = oDialog;
                    this.base.getView().addDependent(this._oReportDialog);
                    
                    var oModel = new JSONModel({
                        taskId: sTaskId,
                        taskName: sTaskName,
                        reportType: "EXECUTIVE",
                        format: "PDF",
                        includeDetails: true,
                        includeScreenshots: true,
                        includeMetrics: true,
                        includeRecommendations: true,
                        distribution: []
                    });
                    this._oReportDialog.setModel(oModel, "report");
                    this._oReportDialog.open();
                }.bind(this));
            } else {
                this._oReportDialog.open();
            }
        },

        onGenerateReport: function() {
            var oModel = this._oReportDialog.getModel("report");
            var oData = oModel.getData();
            
            this._oReportDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent5/v1/tasks/" + oData.taskId + "/generate-report",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    type: oData.reportType,
                    format: oData.format,
                    includeDetails: oData.includeDetails,
                    includeScreenshots: oData.includeScreenshots,
                    includeMetrics: oData.includeMetrics,
                    includeRecommendations: oData.includeRecommendations,
                    distribution: oData.distribution
                }),
                success: function(data) {
                    this._oReportDialog.setBusy(false);
                    this._oReportDialog.close();
                    
                    MessageBox.success(
                        "Test report generated successfully!",
                        {
                            actions: ["Download", "Email", MessageBox.Action.CLOSE],
                            onClose: function(oAction) {
                                if (oAction === "Download") {
                                    window.open(data.downloadUrl, "_blank");
                                } else if (oAction === "Email") {
                                    this._emailReport(data.reportId);
                                }
                            }.bind(this)
                        }
                    );
                }.bind(this),
                error: function(xhr) {
                    this._oReportDialog.setBusy(false);
                    MessageBox.error("Report generation failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        _emailReport: function(sReportId) {
            // Show email distribution dialog
            MessageBox.information("Email distribution functionality will be available soon.");
        },

        onCreateDefect: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            var iFailedTests = oContext.getProperty("failedTests");
            
            if (iFailedTests === 0) {
                MessageBox.information("No failed tests found to create defects from.");
                return;
            }
            
            if (!this._oDefectDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent5.ext.fragment.CreateDefect",
                    controller: this
                }).then(function(oDialog) {
                    this._oDefectDialog = oDialog;
                    this.base.getView().addDependent(this._oDefectDialog);
                    
                    this._loadFailedTests(sTaskId);
                    this._oDefectDialog.open();
                }.bind(this));
            } else {
                this._loadFailedTests(sTaskId);
                this._oDefectDialog.open();
            }
        },

        _loadFailedTests: function(sTaskId) {
            this._oDefectDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent5/v1/tasks/" + sTaskId + "/failed-tests",
                type: "GET",
                success: function(data) {
                    this._oDefectDialog.setBusy(false);
                    
                    var oModel = new JSONModel({
                        taskId: sTaskId,
                        failedTests: data.tests,
                        selectedTests: [],
                        defectInfo: {
                            title: "",
                            description: "",
                            severity: "MEDIUM",
                            priority: "MEDIUM",
                            assignee: "",
                            component: "",
                            labels: []
                        }
                    });
                    this._oDefectDialog.setModel(oModel, "defect");
                }.bind(this),
                error: function(xhr) {
                    this._oDefectDialog.setBusy(false);
                    MessageBox.error("Failed to load failed tests: " + xhr.responseText);
                }.bind(this)
            });
        },

        onSubmitDefect: function() {
            var oModel = this._oDefectDialog.getModel("defect");
            var oData = oModel.getData();
            
            if (oData.selectedTests.length === 0) {
                MessageBox.error("Please select at least one failed test");
                return;
            }
            
            if (!oData.defectInfo.title || !oData.defectInfo.description) {
                MessageBox.error("Please provide defect title and description");
                return;
            }
            
            this._oDefectDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent5/v1/defects",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    taskId: oData.taskId,
                    failedTests: oData.selectedTests,
                    defectInfo: oData.defectInfo
                }),
                success: function(data) {
                    this._oDefectDialog.setBusy(false);
                    this._oDefectDialog.close();
                    
                    MessageBox.success(
                        "Defect created successfully!\n" +
                        "Defect ID: " + data.defectId + "\n" +
                        "Tracking URL: " + data.trackingUrl
                    );
                }.bind(this),
                error: function(xhr) {
                    this._oDefectDialog.setBusy(false);
                    MessageBox.error("Defect creation failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        onScheduleRegression: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            
            if (!this._oRegressionDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent5.ext.fragment.ScheduleRegression",
                    controller: this
                }).then(function(oDialog) {
                    this._oRegressionDialog = oDialog;
                    this.base.getView().addDependent(this._oRegressionDialog);
                    
                    var oModel = new JSONModel({
                        taskId: sTaskId,
                        scheduleType: "IMMEDIATE",
                        cronExpression: "",
                        selectedDate: new Date(),
                        selectedTime: "00:00",
                        includeNewTests: true,
                        onlyFailedTests: false,
                        notifications: true
                    });
                    this._oRegressionDialog.setModel(oModel, "regression");
                    this._oRegressionDialog.open();
                }.bind(this));
            } else {
                this._oRegressionDialog.open();
            }
        },

        onScheduleRegressionExecution: function() {
            var oModel = this._oRegressionDialog.getModel("regression");
            var oData = oModel.getData();
            
            this._oRegressionDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent5/v1/tasks/" + oData.taskId + "/schedule-regression",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    scheduleType: oData.scheduleType,
                    cronExpression: oData.cronExpression,
                    scheduledDateTime: oData.selectedDate + "T" + oData.selectedTime,
                    includeNewTests: oData.includeNewTests,
                    onlyFailedTests: oData.onlyFailedTests,
                    notifications: oData.notifications
                }),
                success: function(data) {
                    this._oRegressionDialog.setBusy(false);
                    this._oRegressionDialog.close();
                    
                    MessageBox.success(
                        "Regression test scheduled successfully!\n" +
                        "Schedule ID: " + data.scheduleId + "\n" +
                        "Next execution: " + data.nextExecution
                    );
                }.bind(this),
                error: function(xhr) {
                    this._oRegressionDialog.setBusy(false);
                    MessageBox.error("Scheduling failed: " + xhr.responseText);
                }.bind(this)
            });
        }
    });
});