sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel"
], function (ControllerExtension, MessageBox, MessageToast, Fragment, JSONModel) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent6.ext.controller.ObjectPageExt", {
        
        override: {
            onInit: function () {
                this._extensionAPI = this.base.getExtensionAPI();
            }
        },

        onStartAssessment: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            var sTaskName = oContext.getProperty("taskName");
            
            MessageBox.confirm("Start quality assessment for '" + sTaskName + "'?", {
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        this._startQualityAssessment(sTaskId);
                    }
                }.bind(this)
            });
        },

        _startQualityAssessment: function(sTaskId) {
            this._extensionAPI.getView().setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent6/v1/tasks/" + sTaskId + "/assess",
                type: "POST",
                success: function(data) {
                    this._extensionAPI.getView().setBusy(false);
                    MessageToast.show("Quality assessment started");
                    this._extensionAPI.refresh();
                    
                    this._startRealtimeMonitoring(sTaskId);
                }.bind(this),
                error: function(xhr) {
                    this._extensionAPI.getView().setBusy(false);
                    MessageBox.error("Failed to start assessment: " + xhr.responseText);
                }.bind(this)
            });
        },

        _startRealtimeMonitoring: function(sTaskId) {
            this._eventSource = new EventSource("/a2a/agent6/v1/tasks/" + sTaskId + "/stream");
            
            this._eventSource.onmessage = function(event) {
                var data = JSON.parse(event.data);
                
                if (data.type === "progress") {
                    this._updateAssessmentProgress(data);
                } else if (data.type === "complete") {
                    this._eventSource.close();
                    this._extensionAPI.refresh();
                    MessageBox.success("Quality assessment completed successfully!");
                    this._showAssessmentResults(data.results);
                } else if (data.type === "error") {
                    this._eventSource.close();
                    MessageBox.error("Assessment failed: " + data.error);
                }
            }.bind(this);
            
            this._eventSource.onerror = function() {
                this._eventSource.close();
                MessageBox.error("Lost connection to assessment process");
            }.bind(this);
        },

        _updateAssessmentProgress: function(data) {
            MessageToast.show(data.component + ": " + data.progress + "%");
        },

        _showAssessmentResults: function(results) {
            var oView = this.base.getView();
            
            if (!this._oResultsDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent6.ext.fragment.AssessmentResults",
                    controller: this
                }).then(function(oDialog) {
                    this._oResultsDialog = oDialog;
                    oView.addDependent(this._oResultsDialog);
                    
                    var oModel = new JSONModel(results);
                    this._oResultsDialog.setModel(oModel, "results");
                    this._oResultsDialog.open();
                    
                    this._createResultsVisualizations(results);
                }.bind(this));
            } else {
                var oModel = new JSONModel(results);
                this._oResultsDialog.setModel(oModel, "results");
                this._oResultsDialog.open();
                this._createResultsVisualizations(results);
            }
        },

        _createResultsVisualizations: function(results) {
            this._createQualityScoreRadar(results.scores);
            this._createIssueBreakdown(results.issues);
        },

        _createQualityScoreRadar: function(scores) {
            var oRadarChart = this._oResultsDialog.byId("qualityRadarChart");
            if (!oRadarChart || !scores) return;
            
            var aRadarData = Object.keys(scores).map(function(key) {
                return {
                    Component: key,
                    Score: scores[key]
                };
            });
            
            var oChartModel = new sap.ui.model.json.JSONModel({
                radarData: aRadarData
            });
            oRadarChart.setModel(oChartModel);
        },

        onMakeRoutingDecision: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            var oQualityScore = oContext.getProperty("overallQuality");
            
            if (!this._oRoutingDecisionDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent6.ext.fragment.RoutingDecisionDialog",
                    controller: this
                }).then(function(oDialog) {
                    this._oRoutingDecisionDialog = oDialog;
                    this.base.getView().addDependent(this._oRoutingDecisionDialog);
                    
                    var oModel = new JSONModel({
                        taskId: sTaskId,
                        currentQuality: oQualityScore,
                        decision: "",
                        targetAgent: "",
                        confidence: 0,
                        reason: "",
                        priority: "NORMAL"
                    });
                    this._oRoutingDecisionDialog.setModel(oModel, "routing");
                    this._oRoutingDecisionDialog.open();
                    
                    this._loadRoutingOptions(sTaskId);
                }.bind(this));
            } else {
                this._oRoutingDecisionDialog.open();
                this._loadRoutingOptions(sTaskId);
            }
        },

        _loadRoutingOptions: function(sTaskId) {
            jQuery.ajax({
                url: "/a2a/agent6/v1/tasks/" + sTaskId + "/routing-options",
                type: "GET",
                success: function(data) {
                    var oModel = this._oRoutingDecisionDialog.getModel("routing");
                    var oData = oModel.getData();
                    
                    oData.availableAgents = data.agents;
                    oData.recommendations = data.recommendations;
                    oData.routingHistory = data.history;
                    
                    oModel.setData(oData);
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to load routing options: " + xhr.responseText);
                }.bind(this)
            });
        },

        onConfirmRoutingDecision: function() {
            var oModel = this._oRoutingDecisionDialog.getModel("routing");
            var oData = oModel.getData();
            
            if (!oData.decision || !oData.targetAgent) {
                MessageBox.error("Please select decision and target agent");
                return;
            }
            
            this._oRoutingDecisionDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent6/v1/tasks/" + oData.taskId + "/route",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    decision: oData.decision,
                    targetAgent: oData.targetAgent,
                    confidence: oData.confidence,
                    reason: oData.reason,
                    priority: oData.priority
                }),
                success: function(data) {
                    this._oRoutingDecisionDialog.setBusy(false);
                    this._oRoutingDecisionDialog.close();
                    
                    MessageBox.success(
                        "Routing decision made successfully!\\n" +
                        "Task routed to: " + oData.targetAgent + "\\n" +
                        "Estimated processing time: " + data.estimatedTime + " minutes"
                    );
                    
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    this._oRoutingDecisionDialog.setBusy(false);
                    MessageBox.error("Failed to make routing decision: " + xhr.responseText);
                }.bind(this)
            });
        },

        onVerifyTrust: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            
            this._extensionAPI.getView().setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent6/v1/tasks/" + sTaskId + "/verify-trust",
                type: "POST",
                success: function(data) {
                    this._extensionAPI.getView().setBusy(false);
                    this._showTrustVerificationResults(data);
                }.bind(this),
                error: function(xhr) {
                    this._extensionAPI.getView().setBusy(false);
                    MessageBox.error("Trust verification failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        _showTrustVerificationResults: function(verificationData) {
            var oView = this.base.getView();
            
            if (!this._oTrustResultsDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent6.ext.fragment.TrustVerificationResults",
                    controller: this
                }).then(function(oDialog) {
                    this._oTrustResultsDialog = oDialog;
                    oView.addDependent(this._oTrustResultsDialog);
                    
                    var oModel = new JSONModel(verificationData);
                    this._oTrustResultsDialog.setModel(oModel, "trust");
                    this._oTrustResultsDialog.open();
                    
                    this._createTrustVisualizations(verificationData);
                }.bind(this));
            } else {
                var oModel = new JSONModel(verificationData);
                this._oTrustResultsDialog.setModel(oModel, "trust");
                this._oTrustResultsDialog.open();
                this._createTrustVisualizations(verificationData);
            }
        },

        _createTrustVisualizations: function(data) {
            var oTrustChart = this._oTrustResultsDialog.byId("trustFactorsChart");
            if (!oTrustChart || !data.factors) return;
            
            var aChartData = data.factors.map(function(factor) {
                return {
                    Factor: factor.name,
                    Score: factor.score,
                    Weight: factor.weight
                };
            });
            
            var oChartModel = new sap.ui.model.json.JSONModel({
                factorData: aChartData
            });
            oTrustChart.setModel(oChartModel);
        },

        onGenerateQualityReport: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            var sTaskName = oContext.getProperty("taskName");
            
            if (!this._oReportDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent6.ext.fragment.QualityReport",
                    controller: this
                }).then(function(oDialog) {
                    this._oReportDialog = oDialog;
                    this.base.getView().addDependent(this._oReportDialog);
                    
                    var oModel = new JSONModel({
                        taskId: sTaskId,
                        taskName: sTaskName,
                        reportType: "COMPREHENSIVE",
                        includeCharts: true,
                        includeRecommendations: true,
                        format: "PDF"
                    });
                    this._oReportDialog.setModel(oModel, "report");
                    this._oReportDialog.open();
                }.bind(this));
            } else {
                this._oReportDialog.open();
            }
        },

        onExecuteReportGeneration: function() {
            var oModel = this._oReportDialog.getModel("report");
            var oData = oModel.getData();
            
            this._oReportDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent6/v1/tasks/" + oData.taskId + "/report",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    reportType: oData.reportType,
                    includeCharts: oData.includeCharts,
                    includeRecommendations: oData.includeRecommendations,
                    format: oData.format
                }),
                success: function(data) {
                    this._oReportDialog.setBusy(false);
                    this._oReportDialog.close();
                    
                    MessageBox.information(
                        "Quality report generated successfully!\\n\\nDownload: " + data.filename,
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

        onOptimizeWorkflow: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            
            MessageBox.confirm(
                "Analyze and optimize workflow for this task? This will identify bottlenecks and suggest improvements.",
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._optimizeWorkflow(sTaskId);
                        }
                    }.bind(this)
                }
            );
        },

        _optimizeWorkflow: function(sTaskId) {
            jQuery.ajax({
                url: "/a2a/agent6/v1/tasks/" + sTaskId + "/optimize",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    analysisDepth: "COMPREHENSIVE",
                    includeResourceOptimization: true,
                    applyOptimizations: false
                }),
                success: function(data) {
                    this._showOptimizationResults(data);
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Workflow optimization failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        _showOptimizationResults: function(optimizationData) {
            var sMessage = "Workflow Analysis Results:\\n\\n";
            
            if (optimizationData.bottlenecks && optimizationData.bottlenecks.length > 0) {
                sMessage += "Bottlenecks Found:\\n";
                optimizationData.bottlenecks.forEach(function(bottleneck) {
                    sMessage += "• " + bottleneck.stage + ": " + bottleneck.impact + "\\n";
                });
                sMessage += "\\n";
            }
            
            if (optimizationData.recommendations && optimizationData.recommendations.length > 0) {
                sMessage += "Optimization Recommendations:\\n";
                optimizationData.recommendations.forEach(function(rec) {
                    sMessage += "• " + rec.description + " (Impact: " + rec.expectedImprovement + ")\\n";
                });
            }
            
            MessageBox.information(sMessage, {
                actions: ["Apply Optimizations", MessageBox.Action.CLOSE],
                onClose: function(oAction) {
                    if (oAction === "Apply Optimizations") {
                        this._applyOptimizations(optimizationData.optimizations);
                    }
                }.bind(this)
            });
        },

        _applyOptimizations: function(optimizations) {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            
            jQuery.ajax({
                url: "/a2a/agent6/v1/tasks/" + sTaskId + "/apply-optimizations",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({ optimizations: optimizations }),
                success: function(data) {
                    MessageBox.success("Workflow optimizations applied successfully!");
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to apply optimizations: " + xhr.responseText);
                }.bind(this)
            });
        },

        onEscalateIssues: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            var aIssues = oContext.getProperty("issuesFound");
            
            if (!aIssues || aIssues.length === 0) {
                MessageBox.information("No issues found to escalate.");
                return;
            }
            
            MessageBox.confirm(
                "Escalate " + aIssues.length + " issues for immediate attention?",
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._escalateIssues(sTaskId, aIssues);
                        }
                    }.bind(this)
                }
            );
        },

        _escalateIssues: function(sTaskId, aIssues) {
            jQuery.ajax({
                url: "/a2a/agent6/v1/tasks/" + sTaskId + "/escalate",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    issues: aIssues,
                    priority: "HIGH",
                    notifyStakeholders: true
                }),
                success: function(data) {
                    MessageBox.success(
                        "Issues escalated successfully!\\n" +
                        "Escalation ID: " + data.escalationId + "\\n" +
                        "Stakeholders notified: " + data.notifiedCount
                    );
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to escalate issues: " + xhr.responseText);
                }.bind(this)
            });
        },

        onViewQualityMetrics: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            
            jQuery.ajax({
                url: "/a2a/agent6/v1/tasks/" + sTaskId + "/metrics",
                type: "GET",
                success: function(data) {
                    this._showQualityMetrics(data);
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to load quality metrics: " + xhr.responseText);
                }.bind(this)
            });
        },

        _showQualityMetrics: function(metricsData) {
            var sMessage = "Quality Metrics Summary:\\n\\n";
            
            Object.keys(metricsData).forEach(function(metric) {
                var value = metricsData[metric];
                sMessage += metric + ": " + value + "\\n";
            });
            
            MessageBox.information(sMessage);
        }
    });
});