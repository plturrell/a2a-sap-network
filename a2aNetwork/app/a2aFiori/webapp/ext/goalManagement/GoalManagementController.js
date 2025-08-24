sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast",
    "sap/m/MessageBox"
], (Controller, JSONModel, MessageToast, MessageBox) => {
    "use strict";

    return Controller.extend("a2a.network.fiori.ext.goalManagement.GoalManagementController", {

        // Formatter functions for UI state management
        formatProgressState(progress) {
            if (progress >= 90) {return "Success";}
            if (progress >= 70) {return "Information";}
            if (progress >= 50) {return "Warning";}
            return "Error";
        },

        formatGoalTypeState(goalType) {
            const stateMap = {
                "performance": "Information",
                "quality": "Success",
                "reliability": "Warning",
                "compliance": "Error"
            };
            return stateMap[goalType] || "None";
        },

        formatPriorityState(priority) {
            const stateMap = {
                "critical": "Error",
                "high": "Warning",
                "medium": "Information",
                "low": "Success"
            };
            return stateMap[priority] || "None";
        },

        formatStatusState(status) {
            const stateMap = {
                "active": "Success",
                "paused": "Warning",
                "completed": "Information",
                "cancelled": "Error",
                "draft": "None"
            };
            return stateMap[status] || "None";
        },

        formatRiskState(riskLevel) {
            const stateMap = {
                "low": "Success",
                "medium": "Warning",
                "high": "Error",
                "critical": "Error"
            };
            return stateMap[riskLevel] || "None";
        },

        formatTrendIcon(trend) {
            const iconMap = {
                "increasing": "sap-icon://trend-up",
                "decreasing": "sap-icon://trend-down",
                "stable": "sap-icon://horizontal-bar-chart"
            };
            return iconMap[trend] || "sap-icon://horizontal-bar-chart";
        },

        formatTrendColor(trend) {
            const colorMap = {
                "increasing": "#2ECC71",
                "decreasing": "#E74C3C",
                "stable": "#F39C12"
            };
            return colorMap[trend] || "#95A5A6";
        },

        formatDate(dateString) {
            if (!dateString) {return "";}
            const date = new Date(dateString);
            return date.toLocaleDateString();
        },

        formatDateTime(dateString) {
            if (!dateString) {return "";}
            const date = new Date(dateString);
            return date.toLocaleString();
        },

        formatMinDate() {
            return new Date();
        },

        // Goal creation dialog handlers
        onCreateGoalConfirm() {
            const oCreateModel = this._oCreateGoalDialog.getModel("createGoal");
            const oGoalData = oCreateModel.getData();

            // Validate required fields
            if (!oGoalData.agentId || !oGoalData.specific || !oGoalData.targetDate) {
                MessageBox.error("Please fill in all required fields.");
                return;
            }

            // Validate measurable targets
            const measurable = oGoalData.measurable || {};
            if (Object.keys(measurable).length === 0) {
                MessageBox.error("Please define at least one measurable target.");
                return;
            }

            this._createGoalViaA2A(oGoalData);
        },

        _createGoalViaA2A(goalData) {
            const that = this;

            // Prepare A2A goal assignment payload
            const a2aPayload = {
                operation: "goal_assignment",
                data: {
                    agent_id: goalData.agentId,
                    goal_id: this._generateGoalId(),
                    goal_type: goalData.goalType,
                    specific: goalData.specific,
                    measurable: goalData.measurable,
                    achievable: goalData.achievable,
                    relevant: goalData.relevant,
                    time_bound: goalData.targetDate,
                    priority: goalData.priority,
                    tracking_frequency: goalData.trackingFrequency,
                    ai_options: goalData.aiOptions || {}
                }
            };

            // Send to CAP service which will forward to A2A backend
            this._callCAPService("/goal-management/Goals", "POST", a2aPayload)
                .then((response) => {
                    MessageToast.show("SMART Goal created and assigned successfully!");
                    that._oCreateGoalDialog.close();
                    that._loadGoalData(); // Refresh dashboard
                })
                .catch((error) => {
                    MessageBox.error(`Failed to create goal: ${ error.message}`);
                });
        },

        _generateGoalId() {
            return `goal_${ Date.now() }_${ Math.random().toString(36).substr(2, 9)}`;
        },

        onCreateGoalCancel() {
            this._oCreateGoalDialog.close();
        },

        // Agent details dialog handlers
        onCloseAgentDetails() {
            this._oAgentDetailsDialog.close();
        },

        onUpdateGoal() {
            const oDetailModel = this._oAgentDetailsDialog.getModel("agentDetail");
            const oAgent = oDetailModel.getData();

            MessageBox.confirm(
                `Update goal for ${ oAgent.agentName }?`,
                {
                    onClose: function(sAction) {
                        if (sAction === MessageBox.Action.OK) {
                            this._updateAgentGoal(oAgent);
                        }
                    }.bind(this)
                }
            );
        },

        _updateAgentGoal(agent) {
            const that = this;

            const updatePayload = {
                operation: "goal_update",
                data: {
                    agent_id: agent.agentId,
                    goal_id: agent.currentGoal?.goal_id,
                    updates: {
                        priority: agent.priority,
                        measurable: agent.measurableTargets,
                        target_date: agent.targetDate
                    }
                }
            };

            this._callCAPService(`/goal-management/Goals(${ agent.currentGoal?.goal_id })`, "PATCH", updatePayload)
                .then((response) => {
                    MessageToast.show("Goal updated successfully!");
                    that._oAgentDetailsDialog.close();
                    that._loadGoalData();
                })
                .catch((error) => {
                    MessageBox.error(`Failed to update goal: ${ error.message}`);
                });
        },

        // CAP service integration
        _callCAPService(endpoint, method, data) {
            return new Promise((resolve, reject) => {
                const settings = {
                    url: endpoint,
                    method,
                    headers: {
                        "Content-Type": "application/json",
                        "X-CSRF-Token": this._getCSRFToken()
                    }
                };

                if (data) {
                    settings.data = JSON.stringify(data);
                }

                jQuery.ajax(settings)
                    .done((response) => {
                        resolve(response);
                    })
                    .fail((xhr, status, error) => {
                        reject(new Error(error || "CAP service call failed"));
                    });
            });
        },

        _getCSRFToken() {
            // Get CSRF token for CAP service calls
            return jQuery.sap.getObject("sap-ui-config.csrf.token") || "fetch";
        },

        // Real-time data management
        _setupWebSocketConnection() {
            // Set up WebSocket for real-time goal updates
            let wsUrl = window.location.protocol === "https:" ? "wss:" : "ws:";
            wsUrl += `//${ window.location.host }/ws/goal-updates`;

            try {
                this._websocket = new WebSocket(wsUrl);

                this._websocket.onmessage = function(event) {
                    const update = JSON.parse(event.data);
                    this._handleRealTimeUpdate(update);
                }.bind(this);

                this._websocket.onerror = function(error) {
                    // WebSocket connection failed, falling back to polling
                    this._setupPolling();
                }.bind(this);

            } catch (error) {
                // WebSocket not supported, using polling
                this._setupPolling();
            }
        },

        _handleRealTimeUpdate(update) {
            const _oModel = this.getView().getModel("goalModel");

            switch (update.type) {
            case "progress_update":
                this._updateAgentProgress(update.agent_id, update.data);
                break;
            case "milestone_achieved":
                this._showMilestoneNotification(update.agent_id, update.data);
                break;
            case "goal_completed":
                this._handleGoalCompletion(update.agent_id, update.data);
                break;
            case "system_analytics":
                this._updateSystemMetrics(update.data);
                break;
            }
        },

        _updateAgentProgress(agentId, progressData) {
            const oModel = this.getView().getModel("goalModel");
            const agents = oModel.getProperty("/agents");

            const agentIndex = agents.findIndex((agent) => {
                return agent.agentId === agentId;
            });

            if (agentIndex >= 0) {
                agents[agentIndex].overallProgress = progressData.overall_progress;
                agents[agentIndex].lastUpdated = new Date().toISOString();
                oModel.setProperty("/agents", agents);

                // Update charts
                this._updateAgentPerformanceChart(agents);
            }
        },

        _showMilestoneNotification(agentId, milestoneData) {
            MessageToast.show(
                `🎉 Milestone achieved by ${ agentId }: ${ milestoneData.title}`,
                { duration: 5000 }
            );
        },

        _handleGoalCompletion(agentId, goalData) {
            MessageBox.success(
                `Goal completed by ${ agentId }!\n\n${ goalData.title}`,
                {
                    title: "Goal Achievement",
                    onClose: function() {
                        this._loadGoalData(); // Refresh all data
                    }.bind(this)
                }
            );
        },

        _updateSystemMetrics(analyticsData) {
            const oModel = this.getView().getModel("goalModel");
            oModel.setProperty("/systemMetrics", analyticsData);
            this._updateProgressChart(analyticsData);
        },

        _setupPolling() {
            // Fallback polling mechanism
            this._pollingInterval = setInterval(() => {
                this._loadGoalData();
            }, 30000); // Poll every 30 seconds
        },

        // Cleanup
        onExit() {
            if (this._websocket) {
                this._websocket.close();
            }

            if (this._pollingInterval) {
                clearInterval(this._pollingInterval);
            }

            if (this._updateInterval) {
                clearInterval(this._updateInterval);
            }
        }
    });
});
