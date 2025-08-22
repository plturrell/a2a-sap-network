sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/ui/core/Fragment",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/model/json/JSONModel"
], function (ControllerExtension, Fragment, MessageBox, MessageToast, JSONModel) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent9.ext.controller.ListReportExt", {
        
        override: {
            onInit: function () {
                this._extensionAPI = this.base.getExtensionAPI();
                this._startRealtimeUpdates();
            }
        },

        onCreateReasoningTask: function() {
            var oView = this.base.getView();
            
            if (!this._oCreateDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent9.ext.fragment.CreateReasoningTask",
                    controller: this
                }).then(function(oDialog) {
                    this._oCreateDialog = oDialog;
                    oView.addDependent(this._oCreateDialog);
                    
                    var oModel = new JSONModel({
                        taskName: "",
                        description: "",
                        reasoningType: "DEDUCTIVE",
                        problemDomain: "",
                        reasoningEngine: "FORWARD_CHAINING",
                        priority: "MEDIUM",
                        confidenceThreshold: 0.85,
                        maxInferenceDepth: 10,
                        chainingStrategy: "BREADTH_FIRST",
                        uncertaintyHandling: "PROBABILISTIC",
                        parallelReasoning: true
                    });
                    this._oCreateDialog.setModel(oModel, "create");
                    this._oCreateDialog.open();
                    this._loadReasoningOptions();
                }.bind(this));
            } else {
                this._oCreateDialog.open();
                this._loadReasoningOptions();
            }
        },

        _loadReasoningOptions: function() {
            jQuery.ajax({
                url: "/a2a/agent9/v1/reasoning-options",
                type: "GET",
                success: function(data) {
                    var oModel = this._oCreateDialog.getModel("create");
                    var oData = oModel.getData();
                    oData.availableEngines = data.engines;
                    oData.problemDomains = data.domains;
                    oData.reasoningTypes = data.types;
                    oModel.setData(oData);
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to load reasoning options: " + xhr.responseText);
                }.bind(this)
            });
        },

        onReasoningDashboard: function() {
            var oView = this.base.getView();
            
            if (!this._oDashboard) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent9.ext.fragment.ReasoningDashboard",
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
                url: "/a2a/agent9/v1/dashboard",
                type: "GET",
                success: function(data) {
                    this._oDashboard.setBusy(false);
                    
                    var oDashboardModel = new JSONModel({
                        summary: data.summary,
                        reasoningMetrics: data.reasoningMetrics,
                        knowledgeBase: data.knowledgeBase,
                        enginePerformance: data.enginePerformance,
                        inferenceTrends: data.inferenceTrends,
                        decisionAccuracy: data.decisionAccuracy
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
            this._createConfidenceTrendsChart(data.inferenceTrends);
            this._createEnginePerformanceChart(data.enginePerformance);
            this._createDecisionAccuracyChart(data.decisionAccuracy);
        },

        _createConfidenceTrendsChart: function(trendsData) {
            var oVizFrame = this._oDashboard.byId("confidenceTrendsChart");
            if (!oVizFrame || !trendsData) return;
            
            var aChartData = trendsData.map(function(trend) {
                return {
                    Time: trend.timestamp,
                    Confidence: trend.averageConfidence,
                    Inferences: trend.inferencesGenerated,
                    Accuracy: trend.accuracy
                };
            });
            
            var oChartModel = new sap.ui.model.json.JSONModel({
                trendsData: aChartData
            });
            oVizFrame.setModel(oChartModel);
            
            oVizFrame.setVizProperties({
                categoryAxis: {
                    title: { text: "Time" }
                },
                valueAxis: {
                    title: { text: "Confidence %" }
                },
                title: {
                    text: "Reasoning Confidence Trends"
                }
            });
        },

        onKnowledgeManager: function() {
            var oView = this.base.getView();
            
            if (!this._oKnowledgeDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent9.ext.fragment.KnowledgeManager",
                    controller: this
                }).then(function(oDialog) {
                    this._oKnowledgeDialog = oDialog;
                    oView.addDependent(this._oKnowledgeDialog);
                    this._oKnowledgeDialog.open();
                    this._loadKnowledgeData();
                }.bind(this));
            } else {
                this._oKnowledgeDialog.open();
                this._loadKnowledgeData();
            }
        },

        _loadKnowledgeData: function() {
            jQuery.ajax({
                url: "/a2a/agent9/v1/knowledge-base",
                type: "GET",
                success: function(data) {
                    var oModel = new JSONModel({
                        facts: data.facts,
                        rules: data.rules,
                        ontologies: data.ontologies,
                        consistency: data.consistency,
                        completeness: data.completeness,
                        domains: data.domains
                    });
                    this._oKnowledgeDialog.setModel(oModel, "knowledge");
                    this._createKnowledgeVisualizations(data);
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to load knowledge data: " + xhr.responseText);
                }.bind(this)
            });
        },

        _createKnowledgeVisualizations: function(data) {
            var oKnowledgeChart = this._oKnowledgeDialog.byId("knowledgeGrowthChart");
            if (!oKnowledgeChart || !data.growth) return;
            
            var aChartData = data.growth.map(function(point) {
                return {
                    Date: point.date,
                    Facts: point.factsCount,
                    Rules: point.rulesCount,
                    Inferences: point.inferencesCount
                };
            });
            
            var oChartModel = new sap.ui.model.json.JSONModel({
                growthData: aChartData
            });
            oKnowledgeChart.setModel(oChartModel);
        },

        onRuleEngine: function() {
            var oView = this.base.getView();
            
            if (!this._oRuleDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent9.ext.fragment.RuleEngine",
                    controller: this
                }).then(function(oDialog) {
                    this._oRuleDialog = oDialog;
                    oView.addDependent(this._oRuleDialog);
                    this._oRuleDialog.open();
                    this._loadRuleData();
                }.bind(this));
            } else {
                this._oRuleDialog.open();
                this._loadRuleData();
            }
        },

        _loadRuleData: function() {
            jQuery.ajax({
                url: "/a2a/agent9/v1/rules",
                type: "GET",
                success: function(data) {
                    var oModel = new JSONModel({
                        rules: data.rules,
                        ruleTypes: data.ruleTypes,
                        conflictResolution: data.conflictResolution,
                        rulePerformance: data.performance
                    });
                    this._oRuleDialog.setModel(oModel, "rules");
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to load rule data: " + xhr.responseText);
                }.bind(this)
            });
        },

        onInferenceEngine: function() {
            var oView = this.base.getView();
            
            if (!this._oInferenceDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent9.ext.fragment.InferenceEngine",
                    controller: this
                }).then(function(oDialog) {
                    this._oInferenceDialog = oDialog;
                    oView.addDependent(this._oInferenceDialog);
                    this._oInferenceDialog.open();
                    this._loadInferenceData();
                }.bind(this));
            } else {
                this._oInferenceDialog.open();
                this._loadInferenceData();
            }
        },

        _loadInferenceData: function() {
            jQuery.ajax({
                url: "/a2a/agent9/v1/inferences",
                type: "GET",
                success: function(data) {
                    var oModel = new JSONModel({
                        inferences: data.inferences,
                        inferenceChains: data.chains,
                        confidence: data.confidence,
                        validation: data.validation
                    });
                    this._oInferenceDialog.setModel(oModel, "inference");
                    this._createInferenceVisualizations(data);
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to load inference data: " + xhr.responseText);
                }.bind(this)
            });
        },

        _createInferenceVisualizations: function(data) {
            // Create inference chain visualization
            this._createInferenceChainDiagram(data.chains);
            this._createConfidenceDistribution(data.confidence);
        },

        _createInferenceChainDiagram: function(chains) {
            // Implementation would create a network diagram showing inference relationships
            // This is a placeholder for the actual visualization logic
        },

        onDecisionMaker: function() {
            var oView = this.base.getView();
            
            if (!this._oDecisionDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent9.ext.fragment.DecisionMaker",
                    controller: this
                }).then(function(oDialog) {
                    this._oDecisionDialog = oDialog;
                    oView.addDependent(this._oDecisionDialog);
                    this._oDecisionDialog.open();
                    this._loadDecisionData();
                }.bind(this));
            } else {
                this._oDecisionDialog.open();
                this._loadDecisionData();
            }
        },

        _loadDecisionData: function() {
            jQuery.ajax({
                url: "/a2a/agent9/v1/decisions",
                type: "GET",
                success: function(data) {
                    var oModel = new JSONModel({
                        decisions: data.decisions,
                        criteria: data.criteria,
                        alternatives: data.alternatives,
                        recommendations: data.recommendations
                    });
                    this._oDecisionDialog.setModel(oModel, "decision");
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to load decision data: " + xhr.responseText);
                }.bind(this)
            });
        },

        onProblemSolver: function() {
            var oView = this.base.getView();
            
            if (!this._oProblemDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent9.ext.fragment.ProblemSolver",
                    controller: this
                }).then(function(oDialog) {
                    this._oProblemDialog = oDialog;
                    oView.addDependent(this._oProblemDialog);
                    this._oProblemDialog.open();
                    this._loadProblemData();
                }.bind(this));
            } else {
                this._oProblemDialog.open();
                this._loadProblemData();
            }
        },

        _loadProblemData: function() {
            jQuery.ajax({
                url: "/a2a/agent9/v1/problems",
                type: "GET",
                success: function(data) {
                    var oModel = new JSONModel({
                        problems: data.problems,
                        solutions: data.solutions,
                        strategies: data.strategies,
                        complexity: data.complexity
                    });
                    this._oProblemDialog.setModel(oModel, "problem");
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to load problem data: " + xhr.responseText);
                }.bind(this)
            });
        },

        onLogicalAnalyzer: function() {
            var oView = this.base.getView();
            
            if (!this._oAnalyzerDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent9.ext.fragment.LogicalAnalyzer",
                    controller: this
                }).then(function(oDialog) {
                    this._oAnalyzerDialog = oDialog;
                    oView.addDependent(this._oAnalyzerDialog);
                    this._oAnalyzerDialog.open();
                    this._loadAnalysisData();
                }.bind(this));
            } else {
                this._oAnalyzerDialog.open();
                this._loadAnalysisData();
            }
        },

        _loadAnalysisData: function() {
            jQuery.ajax({
                url: "/a2a/agent9/v1/analysis",
                type: "GET",
                success: function(data) {
                    var oModel = new JSONModel({
                        contradictions: data.contradictions,
                        consistencyChecks: data.consistencyChecks,
                        logicalErrors: data.logicalErrors,
                        optimization: data.optimization
                    });
                    this._oAnalyzerDialog.setModel(oModel, "analysis");
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to load analysis data: " + xhr.responseText);
                }.bind(this)
            });
        },

        _startRealtimeUpdates: function() {
            this._realtimeEventSource = new EventSource("/a2a/agent9/v1/realtime-updates");
            
            this._realtimeEventSource.onmessage = function(event) {
                var data = JSON.parse(event.data);
                
                if (data.type === "reasoning_complete") {
                    MessageToast.show("Reasoning completed: " + data.taskName);
                    this._extensionAPI.refresh();
                } else if (data.type === "inference_generated") {
                    MessageToast.show("New inference: " + data.inference);
                } else if (data.type === "contradiction_detected") {
                    MessageToast.show("Contradiction detected and resolved");
                }
            }.bind(this);
            
            this._realtimeEventSource.onerror = function() {
                MessageToast.show("Real-time updates disconnected");
            }.bind(this);
        },

        onConfirmCreateTask: function() {
            var oModel = this._oCreateDialog.getModel("create");
            var oData = oModel.getData();
            
            if (!oData.taskName || !oData.reasoningType || !oData.problemDomain) {
                MessageBox.error("Please fill all required fields");
                return;
            }
            
            this._oCreateDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent9/v1/reasoning-tasks",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify(oData),
                success: function(data) {
                    this._oCreateDialog.setBusy(false);
                    this._oCreateDialog.close();
                    MessageToast.show("Reasoning task created successfully");
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