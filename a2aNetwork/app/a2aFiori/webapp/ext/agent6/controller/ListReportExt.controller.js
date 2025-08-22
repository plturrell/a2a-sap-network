sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/ui/core/Fragment",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/model/json/JSONModel"
], function (ControllerExtension, Fragment, MessageBox, MessageToast, JSONModel) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent6.ext.controller.ListReportExt", {
        
        override: {
            onInit: function () {
                this._extensionAPI = this.base.getExtensionAPI();
            }
        },

        onCreateQualityTask: function() {
            var oView = this.base.getView();
            
            if (!this._oCreateDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent6.ext.fragment.CreateQualityTask",
                    controller: this
                }).then(function(oDialog) {
                    this._oCreateDialog = oDialog;
                    oView.addDependent(this._oCreateDialog);
                    
                    var oModel = new JSONModel({
                        taskName: "",
                        description: "",
                        qualityGate: "",
                        dataSource: "",
                        processingPipeline: "",
                        evaluationCriteria: {
                            compliance: true,
                            performance: true,
                            security: true,
                            reliability: true,
                            usability: false,
                            maintainability: false
                        },
                        thresholds: {
                            minQualityScore: 80,
                            maxIssues: 5,
                            minTrustScore: 75
                        }
                    });
                    this._oCreateDialog.setModel(oModel, "create");
                    this._oCreateDialog.open();
                }.bind(this));
            } else {
                this._oCreateDialog.open();
            }
        },

        onQualityDashboard: function() {
            var oView = this.base.getView();
            
            if (!this._oDashboard) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent6.ext.fragment.QualityDashboard",
                    controller: this
                }).then(function(oDialog) {
                    this._oDashboard = oDialog;
                    oView.addDependent(this._oDashboard);
                    this._oDashboard.open();
                    this._loadDashboardData();
                }.bind(this));
            } else {
                this._oDashboard.open();
                this._loadDashboardData();
            }
        },

        _loadDashboardData: function() {
            this._oDashboard.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent6/v1/dashboard",
                type: "GET",
                success: function(data) {
                    this._oDashboard.setBusy(false);
                    
                    var oDashboardModel = new JSONModel({
                        metrics: data.metrics,
                        trends: data.trends,
                        qualityGates: data.qualityGates,
                        routingStats: data.routingStats,
                        trustMetrics: data.trustMetrics,
                        workflowHealth: data.workflowHealth
                    });
                    
                    this._oDashboard.setModel(oDashboardModel, "dashboard");
                    this._createDashboardCharts(data);
                }.bind(this),
                error: function(xhr) {
                    this._oDashboard.setBusy(false);
                    MessageBox.error("Failed to load dashboard data: " + xhr.responseText);
                }.bind(this)
            });
        },

        _createDashboardCharts: function(data) {
            this._createQualityTrendsChart(data.trends);
            this._createRoutingDistributionChart(data.routingStats);
            this._createTrustScoreDistribution(data.trustMetrics);
        },

        _createQualityTrendsChart: function(trends) {
            var oVizFrame = this._oDashboard.byId("qualityTrendsChart");
            if (!oVizFrame || !trends) return;
            
            var aChartData = trends.map(function(trend) {
                return {
                    Date: trend.date,
                    QualityScore: trend.averageQuality,
                    TaskCount: trend.taskCount,
                    IssueCount: trend.issueCount
                };
            });
            
            var oChartModel = new sap.ui.model.json.JSONModel({
                trendData: aChartData
            });
            oVizFrame.setModel(oChartModel);
            
            oVizFrame.setVizProperties({
                categoryAxis: {
                    title: { text: "Date" }
                },
                valueAxis: {
                    title: { text: "Quality Score" }
                },
                title: {
                    text: "Quality Trends Over Time"
                }
            });
        },

        _createRoutingDistributionChart: function(routingStats) {
            var oVizFrame = this._oDashboard.byId("routingDistributionChart");
            if (!oVizFrame || !routingStats) return;
            
            var aChartData = Object.keys(routingStats).map(function(agent) {
                return {
                    Agent: agent,
                    TaskCount: routingStats[agent].count,
                    SuccessRate: routingStats[agent].successRate
                };
            });
            
            var oChartModel = new sap.ui.model.json.JSONModel({
                routingData: aChartData
            });
            oVizFrame.setModel(oChartModel);
        },

        onRoutingDecisionManager: function() {
            var oView = this.base.getView();
            
            if (!this._oRoutingDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent6.ext.fragment.RoutingDecisionManager",
                    controller: this
                }).then(function(oDialog) {
                    this._oRoutingDialog = oDialog;
                    oView.addDependent(this._oRoutingDialog);
                    this._oRoutingDialog.open();
                    this._loadRoutingData();
                }.bind(this));
            } else {
                this._oRoutingDialog.open();
                this._loadRoutingData();
            }
        },

        _loadRoutingData: function() {
            jQuery.ajax({
                url: "/a2a/agent6/v1/routing-rules",
                type: "GET",
                success: function(data) {
                    var oModel = new JSONModel({
                        rules: data.rules,
                        agents: data.availableAgents,
                        pendingDecisions: data.pendingDecisions
                    });
                    this._oRoutingDialog.setModel(oModel, "routing");
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to load routing data: " + xhr.responseText);
                }.bind(this)
            });
        },

        onTrustVerification: function() {
            var oView = this.base.getView();
            
            if (!this._oTrustDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent6.ext.fragment.TrustVerification",
                    controller: this
                }).then(function(oDialog) {
                    this._oTrustDialog = oDialog;
                    oView.addDependent(this._oTrustDialog);
                    this._oTrustDialog.open();
                    this._loadTrustData();
                }.bind(this));
            } else {
                this._oTrustDialog.open();
                this._loadTrustData();
            }
        },

        _loadTrustData: function() {
            jQuery.ajax({
                url: "/a2a/agent6/v1/trust-metrics",
                type: "GET",
                success: function(data) {
                    var oModel = new JSONModel({
                        verificationQueue: data.verificationQueue,
                        trustFactors: data.trustFactors,
                        blockchainStatus: data.blockchainStatus,
                        reputationScores: data.reputationScores
                    });
                    this._oTrustDialog.setModel(oModel, "trust");
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to load trust data: " + xhr.responseText);
                }.bind(this)
            });
        },

        onWorkflowOptimization: function() {
            var oView = this.base.getView();
            
            if (!this._oWorkflowDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent6.ext.fragment.WorkflowOptimization",
                    controller: this
                }).then(function(oDialog) {
                    this._oWorkflowDialog = oDialog;
                    oView.addDependent(this._oWorkflowDialog);
                    this._oWorkflowDialog.open();
                    this._loadWorkflowData();
                }.bind(this));
            } else {
                this._oWorkflowDialog.open();
                this._loadWorkflowData();
            }
        },

        _loadWorkflowData: function() {
            jQuery.ajax({
                url: "/a2a/agent6/v1/workflow-analysis",
                type: "GET",
                success: function(data) {
                    var oModel = new JSONModel({
                        bottlenecks: data.bottlenecks,
                        optimizations: data.optimizations,
                        performance: data.performance,
                        resourceUtilization: data.resourceUtilization
                    });
                    this._oWorkflowDialog.setModel(oModel, "workflow");
                    this._createWorkflowVisualizations(data);
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to load workflow data: " + xhr.responseText);
                }.bind(this)
            });
        },

        _createWorkflowVisualizations: function(data) {
            var oChart = this._oWorkflowDialog.byId("workflowChart");
            if (!oChart) return;
            
            var aChartData = data.performance.map(function(perf) {
                return {
                    Stage: perf.stage,
                    Duration: perf.averageDuration,
                    Efficiency: perf.efficiency
                };
            });
            
            var oChartModel = new sap.ui.model.json.JSONModel({
                performanceData: aChartData
            });
            oChart.setModel(oChartModel);
        },

        onBatchAssessment: function() {
            var oTable = this._extensionAPI.getTable();
            var aSelectedContexts = oTable.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageBox.warning("Please select at least one task for batch assessment.");
                return;
            }
            
            MessageBox.confirm(
                "Start batch quality assessment for " + aSelectedContexts.length + " tasks?",
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._startBatchAssessment(aSelectedContexts);
                        }
                    }.bind(this)
                }
            );
        },

        _startBatchAssessment: function(aContexts) {
            var aTaskIds = aContexts.map(function(oContext) {
                return oContext.getProperty("ID");
            });
            
            this.base.getView().setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent6/v1/batch-assessment",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    taskIds: aTaskIds,
                    assessmentType: "COMPREHENSIVE",
                    parallel: true
                }),
                success: function(data) {
                    this.base.getView().setBusy(false);
                    MessageBox.success(
                        "Batch assessment started!\\n" +
                        "Job ID: " + data.jobId + "\\n" +
                        "Estimated time: " + data.estimatedTime + " minutes"
                    );
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    this.base.getView().setBusy(false);
                    MessageBox.error("Batch assessment failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        onConfigureQualityGates: function() {
            var oRouter = sap.ui.core.UIComponent.getRouterFor(this.base.getView());
            oRouter.navTo("QualityGates");
        },

        onConfirmCreateTask: function() {
            var oModel = this._oCreateDialog.getModel("create");
            var oData = oModel.getData();
            
            if (!oData.taskName || !oData.qualityGate) {
                MessageBox.error("Please fill all required fields");
                return;
            }
            
            this._oCreateDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent6/v1/tasks",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify(oData),
                success: function(data) {
                    this._oCreateDialog.setBusy(false);
                    this._oCreateDialog.close();
                    MessageToast.show("Quality control task created successfully");
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
        }
    });
});