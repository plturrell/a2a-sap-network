sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast",
    "sap/ui/core/Fragment",
    "sap/viz/ui5/controls/VizFrame",
    "sap/viz/ui5/data/FlattenedDataset",
    "sap/viz/ui5/controls/common/feeds/FeedItem"
], (Controller, JSONModel, MessageToast, Fragment, VizFrame, FlattenedDataset, FeedItem) => {
    "use strict";

    return Controller.extend("a2a.network.fiori.ext.goalManagement.GoalDashboard", {

        onInit() {
            this._initializeGoalDashboard();
            this._setupRealTimeUpdates();
        },

        onSyncGoals() {
            const that = this;
            MessageToast.show("Synchronizing goals from orchestrator...");

            // Show busy indicator
            this.getView().setBusy(true);

            // Call sync service
            this._callA2AService("/api/v1/goal-management/syncGoals", "POST")
                .then((oResult) => {
                    that.getView().setBusy(false);

                    if (oResult.status === "success") {
                        const msg = `Goal sync completed: ${oResult.result.successCount} agents synced successfully`;
                        MessageToast.show(msg);

                        // Reload goal data
                        that._loadGoalData();
                    } else {
                        MessageToast.show("Goal sync failed");
                    }
                })
                .catch((oError) => {
                    that.getView().setBusy(false);
                    MessageToast.show(`Sync failed: ${oError.message}`);
                });
        },

        _initializeGoalDashboard() {
            // Initialize goal management data model
            const oGoalModel = new JSONModel({
                agents: [],
                systemMetrics: {
                    totalAgents: 0,
                    averageProgress: 0,
                    activeGoals: 0,
                    completedGoals: 0,
                    milestonesAchieved: 0
                },
                goalTypes: [
                    { key: "performance", text: "Performance Goals" },
                    { key: "quality", text: "Quality Goals" },
                    { key: "reliability", text: "Reliability Goals" },
                    { key: "compliance", text: "Compliance Goals" }
                ],
                priorities: [
                    { key: "critical", text: "Critical", color: "#BB0000" },
                    { key: "high", text: "High", color: "#FF6600" },
                    { key: "medium", text: "Medium", color: "#FFAA00" },
                    { key: "low", text: "Low", color: "#00AA00" }
                ]
            });

            this.getView().setModel(oGoalModel, "goalModel");
            this._loadGoalData();
        },

        _loadGoalData() {
            const that = this;

            // Load goal data from A2A backend
            this._callA2AService("/api/v1/goals/analytics", "GET")
                .then((oData) => {
                    that._processGoalAnalytics(oData);
                })
                .catch((oError) => {
                    MessageToast.show(`Failed to load goal analytics: ${ oError.message}`);
                });

            // Load agent-specific goal data
            this._callA2AService("/api/v1/agents", "GET")
                .then((oData) => {
                    return Promise.all(oData.agents.map((agent) => {
                        return that._loadAgentGoals(agent.agent_id);
                    }));
                })
                .then((agentGoals) => {
                    that._processAgentGoals(agentGoals);
                })
                .catch((oError) => {
                    MessageToast.show(`Failed to load agent goals: ${ oError.message}`);
                });
        },

        _loadAgentGoals(agentId) {
            return this._callA2AService(`/api/v1/goals/agent/${ agentId}`, "GET");
        },

        _processGoalAnalytics(oData) {
            const oModel = this.getView().getModel("goalModel");
            const oSystemMetrics = {
                totalAgents: oData.total_agents_with_goals || 0,
                averageProgress: Math.round(oData.average_progress || 0),
                activeGoals: oData.active_goals || 0,
                completedGoals: oData.completed_goals || 0,
                milestonesAchieved: oData.total_milestones || 0,
                agentsAbove50Percent: oData.agents_above_50_percent || 0
            };

            oModel.setProperty("/systemMetrics", oSystemMetrics);
            this._updateProgressChart(oData);
        },

        _processAgentGoals(agentGoalsArray) {
            const oModel = this.getView().getModel("goalModel");
            const aProcessedAgents = [];

            agentGoalsArray.forEach((agentGoalData) => {
                if (agentGoalData && agentGoalData.agent_id) {
                    const oAgent = {
                        agentId: agentGoalData.agent_id,
                        agentName: agentGoalData.agent_name || agentGoalData.agent_id,
                        totalGoals: agentGoalData.total_goals || 0,
                        overallProgress: Math.round(agentGoalData.overall_progress || 0),
                        status: agentGoalData.status || "active",
                        lastUpdated: agentGoalData.last_updated || new Date().toISOString(),
                        goals: agentGoalData.goals || [],
                        milestones: agentGoalData.milestones_achieved || 0,
                        successCriteriaMet: agentGoalData.success_criteria_met || 0,
                        progressTrend: this._calculateProgressTrend(agentGoalData.progress_history || [])
                    };

                    // Add SMART goal details
                    if (agentGoalData.goals && agentGoalData.goals.length > 0) {
                        oAgent.currentGoal = agentGoalData.goals[0];
                        oAgent.goalType = agentGoalData.goals[0].goal_type || "performance";
                        oAgent.priority = agentGoalData.goals[0].priority || "medium";
                        oAgent.targetDate = agentGoalData.goals[0].target_date;
                        oAgent.measurableTargets = agentGoalData.goals[0].measurable || {};
                    }

                    aProcessedAgents.push(oAgent);
                }
            });

            oModel.setProperty("/agents", aProcessedAgents);
            this._updateAgentPerformanceChart(aProcessedAgents);
        },

        _calculateProgressTrend(progressHistory) {
            if (!progressHistory || progressHistory.length < 2) {
                return "stable";
            }

            const recent = progressHistory.slice(-3);
            const trend = recent[recent.length - 1].progress - recent[0].progress;

            if (trend > 5) {return "increasing";}
            if (trend < -5) {return "decreasing";}
            return "stable";
        },

        _updateProgressChart(oData) {
            const oVizFrame = this.byId("goalProgressChart");
            if (!oVizFrame) {return;}

            const aChartData = [
                { metric: "Active Goals", value: oData.active_goals || 0, color: "#5899DA" },
                { metric: "Completed Goals", value: oData.completed_goals || 0, color: "#E8743B" },
                { metric: "Agents Above 50%", value: oData.agents_above_50_percent || 0, color: "#19A979" },
                { metric: "Total Milestones", value: oData.total_milestones || 0, color: "#ED4A7B" }
            ];

            const oChartModel = new JSONModel({ data: aChartData });
            oVizFrame.setModel(oChartModel);

            oVizFrame.setVizProperties({
                plotArea: {
                    colorPalette: ["#5899DA", "#E8743B", "#19A979", "#ED4A7B"],
                    dataLabel: { visible: true }
                },
                title: { text: "Goal Management Overview" }
            });
        },

        _updateAgentPerformanceChart(aAgents) {
            const oVizFrame = this.byId("agentPerformanceChart");
            if (!oVizFrame) {return;}

            const aChartData = aAgents.map((agent) => {
                return {
                    agentName: agent.agentName,
                    progress: agent.overallProgress,
                    milestones: agent.milestones,
                    goalType: agent.goalType || "performance"
                };
            });

            const oChartModel = new JSONModel({ data: aChartData });
            oVizFrame.setModel(oChartModel);

            oVizFrame.setVizProperties({
                plotArea: {
                    dataLabel: { visible: true },
                    colorPalette: ["#5899DA", "#E8743B", "#19A979", "#ED4A7B", "#945ECF"]
                },
                title: { text: "Agent Goal Progress" },
                valueAxis: { title: { text: "Progress %" } },
                categoryAxis: { title: { text: "Agents" } }
            });
        },

        _setupRealTimeUpdates() {
            // Set up real-time updates every 30 seconds
            this._updateInterval = setInterval(() => {
                this._loadGoalData();
            }, 30000);
        },

        onCreateGoal() {
            const that = this;

            if (!this._oCreateGoalDialog) {
                Fragment.load({
                    name: "a2a.network.fiori.ext.goalManagement.CreateGoalDialog",
                    controller: this
                }).then((oDialog) => {
                    that._oCreateGoalDialog = oDialog;
                    that.getView().addDependent(oDialog);
                    oDialog.open();
                });
            } else {
                this._oCreateGoalDialog.open();
            }
        },

        onAgentGoalPress(oEvent) {
            const oBindingContext = oEvent.getSource().getBindingContext("goalModel");
            const oAgent = oBindingContext.getObject();

            this._showAgentGoalDetails(oAgent);
        },

        _showAgentGoalDetails(oAgent) {
            const that = this;

            if (!this._oAgentDetailsDialog) {
                Fragment.load({
                    name: "a2a.network.fiori.ext.goalManagement.AgentGoalDetails",
                    controller: this
                }).then((oDialog) => {
                    that._oAgentDetailsDialog = oDialog;
                    that.getView().addDependent(oDialog);
                    that._displayAgentDetails(oAgent, oDialog);
                });
            } else {
                this._displayAgentDetails(oAgent, this._oAgentDetailsDialog);
            }
        },

        _displayAgentDetails(oAgent, oDialog) {
            const oDetailModel = new JSONModel(oAgent);
            oDialog.setModel(oDetailModel, "agentDetail");
            oDialog.open();
        },

        onRefreshGoals() {
            MessageToast.show("Refreshing goal data...");
            this._loadGoalData();
        },

        onFilterGoals(oEvent) {
            const sQuery = oEvent.getParameter("query");
            const oTable = this.byId("agentGoalsTable");
            const oBinding = oTable.getBinding("items");

            if (sQuery) {
                const oFilter = new sap.ui.model.Filter([
                    new sap.ui.model.Filter("agentName", sap.ui.model.FilterOperator.Contains, sQuery),
                    new sap.ui.model.Filter("goalType", sap.ui.model.FilterOperator.Contains, sQuery),
                    new sap.ui.model.Filter("status", sap.ui.model.FilterOperator.Contains, sQuery)
                ], false);
                oBinding.filter([oFilter]);
            } else {
                oBinding.filter([]);
            }
        },

        _callA2AService(sPath, sMethod, oData) {
            return new Promise((resolve, reject) => {
                jQuery.ajax({
                    url: sPath,
                    method: sMethod,
                    data: oData ? JSON.stringify(oData) : undefined,
                    contentType: "application/json",
                    headers: {
                        "X-A2A-Agent": "fiori-ui",
                        "Authorization": `Bearer ${this._getA2AToken()}`
                    },
                    success(data) {
                        resolve(data);
                    },
                    error(xhr, status, error) {
                        reject(new Error(error || "Service call failed"));
                    }
                });
            });
        },

        _getA2AToken() {
            // In production, this would get the actual A2A authentication token
            return "a2a-fiori-token";
        },

        onExit() {
            if (this._updateInterval) {
                clearInterval(this._updateInterval);
            }
            if (this._websocket) {
                this._websocket.close();
            }
            if (this._oCreateGoalDialog) {
                this._oCreateGoalDialog.destroy();
            }
            if (this._oAgentDetailsDialog) {
                this._oAgentDetailsDialog.destroy();
            }
        }
    });
});
