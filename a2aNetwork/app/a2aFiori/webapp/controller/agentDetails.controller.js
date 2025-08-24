/**
 * A2A Protocol Compliance: HTTP client usage replaced with blockchain messaging
 */

sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast",
    "sap/ui/core/routing/History",
    "sap/ui/core/format/DateFormat",
    "sap/ui/model/Filter",
    "sap/ui/model/FilterOperator"
], function (Controller, JSONModel, MessageToast, History, DateFormat, Filter, FilterOperator) {
    "use strict";

    return Controller.extend("a2a.network.fiori.controller.agentDetails", {
        
        // Agent metadata matching the server configuration
        AGENT_METADATA: {
            0: { name: "Data Product Agent", port: 8000, type: "Core Processing", icon: "sap-icon://product", color: "#1873B4" },
            1: { name: "Data Standardization", port: 8001, type: "Core Processing", icon: "sap-icon://synchronize", color: "#2C5AA8" },
            2: { name: "AI Preparation", port: 8002, type: "Core Processing", icon: "sap-icon://artificial-intelligence", color: "#5E35B1" },
            3: { name: "Vector Processing", port: 8003, type: "Core Processing", icon: "sap-icon://scatter-chart", color: "#00ACC1" },
            4: { name: "Calc Validation", port: 8004, type: "Core Processing", icon: "sap-icon://validate", color: "#00897B" },
            5: { name: "QA Validation", port: 8005, type: "Core Processing", icon: "sap-icon://quality-issue", color: "#43A047" },
            6: { name: "Quality Control Manager", port: 8006, type: "Management", icon: "sap-icon://process", color: "#E65100" },
            7: { name: "Agent Manager", port: 8007, type: "Management", icon: "sap-icon://org-chart", color: "#D84315" },
            8: { name: "Data Manager", port: 8008, type: "Management", icon: "sap-icon://database", color: "#6D4C41" },
            9: { name: "Reasoning Agent", port: 8009, type: "Management", icon: "sap-icon://decision", color: "#424242" },
            10: { name: "Calculation Agent", port: 8010, type: "Specialized", icon: "sap-icon://sum", color: "#F4511E" },
            11: { name: "SQL Agent", port: 8011, type: "Specialized", icon: "sap-icon://table-view", color: "#C62828" },
            12: { name: "Catalog Manager", port: 8012, type: "Specialized", icon: "sap-icon://course-book", color: "#AD1457" },
            13: { name: "Agent Builder", port: 8013, type: "Specialized", icon: "sap-icon://build", color: "#6A1B9A" },
            14: { name: "Embedding Fine-Tuner", port: 8014, type: "Specialized", icon: "sap-icon://machine-learning", color: "#4527A0" },
            15: { name: "Orchestrator Agent", port: 8015, type: "Specialized", icon: "sap-icon://workflow-tasks", color: "#283593" }
        },

        onInit: function () {
            const oRouter = this.getOwnerComponent().getRouter();
            oRouter.getRoute("agentDetails").attachPatternMatched(this._onAgentMatched, this);

            // Initialize models
            this.getView().setModel(new JSONModel({
                agent: {},
                performance: {},
                tasks: [],
                interactions: [],
                busy: false
            }));

            // Set up refresh interval
            this._refreshInterval = null;
        },

        _onAgentMatched: function (oEvent) {
            const agentId = parseInt(oEvent.getParameter("arguments").agentId);
            this._agentId = agentId;
            
            // Load agent details
            this._loadAgentDetails(agentId);
            
            // Start auto-refresh
            this._startAutoRefresh();
        },

        _loadAgentDetails: function (agentId) {
            const oModel = this.getView().getModel();
            const agent = this.AGENT_METADATA[agentId];
            
            if (!agent) {
                MessageToast.show("Agent not found");
                this.onNavBack();
                return;
            }

            oModel.setProperty("/busy", true);
            
            // Set basic agent info
            oModel.setProperty("/agent", {
                id: agentId,
                name: agent.name,
                type: agent.type,
                port: agent.port,
                icon: agent.icon,
                color: agent.color
            });

            // Fetch agent status
            this._fetchAgentStatus(agentId);
            
            // Fetch agent performance metrics
            this._fetchAgentPerformance(agentId);
            
            // Fetch recent tasks
            this._fetchAgentTasks(agentId);
            
            // Fetch agent interactions
            this._fetchAgentInteractions(agentId);
        },

        _fetchAgentStatus: function (agentId) {
            blockchainClient.sendMessage(`/api/v1/agents/${agentId}/status`)
                .then(response => response.json())
                .then(data => {
                    const oModel = this.getView().getModel();
                    oModel.setProperty("/agent/status", data.d.status);
                    oModel.setProperty("/agent/uptime", data.d.number);
                    oModel.setProperty("/agent/state", data.d.numberState);
                    oModel.setProperty("/busy", false);
                })
                .catch(error => {
                    console.error("Error fetching agent status:", error);
                    this.getView().getModel().setProperty("/busy", false);
                });
        },

        _fetchAgentPerformance: function (agentId) {
            blockchainClient.sendMessage(`/api/v1/agents/${agentId}/performance`)
                .then(response => response.json())
                .then(data => {
                    this.getView().getModel().setProperty("/performance", data.d || {
                        totalTasks: 0,
                        successRate: 0,
                        avgResponseTime: 0,
                        reputation: 0
                    });
                })
                .catch(error => {
                    console.error("Error fetching agent performance:", error);
                });
        },

        _fetchAgentTasks: function (agentId) {
            // Simulated task data - in production, fetch from actual endpoint
            const tasks = [
                { id: 1, name: "Data Processing", status: "completed", duration: "2.3s", timestamp: new Date() },
                { id: 2, name: "Validation Check", status: "completed", duration: "0.8s", timestamp: new Date() },
                { id: 3, name: "Model Training", status: "in_progress", duration: "5m 23s", timestamp: new Date() }
            ];
            
            this.getView().getModel().setProperty("/tasks", tasks);
        },

        _fetchAgentInteractions: function (agentId) {
            // Simulated interaction data - in production, fetch from actual endpoint
            const interactions = [
                { partner: "Agent Manager", type: "Request", message: "Task assignment", time: "2 mins ago" },
                { partner: "Data Manager", type: "Response", message: "Data retrieved", time: "5 mins ago" },
                { partner: "Quality Control", type: "Request", message: "Validation needed", time: "10 mins ago" }
            ];
            
            this.getView().getModel().setProperty("/interactions", interactions);
        },

        _startAutoRefresh: function () {
            // Clear existing interval if any
            if (this._refreshInterval) {
                clearInterval(this._refreshInterval);
            }
            
            // Refresh every 30 seconds
            this._refreshInterval = setInterval(() => {
                this._loadAgentDetails(this._agentId);
            }, 30000);
        },

        onRefresh: function () {
            MessageToast.show("Refreshing agent data...");
            this._loadAgentDetails(this._agentId);
        },

        onStartAgent: function () {
            MessageToast.show(`Starting ${this.AGENT_METADATA[this._agentId].name}...`);
            // Implement actual agent start logic
        },

        onStopAgent: function () {
            MessageToast.show(`Stopping ${this.AGENT_METADATA[this._agentId].name}...`);
            // Implement actual agent stop logic
        },

        onRestartAgent: function () {
            MessageToast.show(`Restarting ${this.AGENT_METADATA[this._agentId].name}...`);
            // Implement actual agent restart logic
        },

        onViewLogs: function () {
            // Navigate to logs view
            this.getOwnerComponent().getRouter().navTo("agentLogs", {
                agentId: this._agentId
            });
        },

        onConfigureAgent: function () {
            // Open configuration dialog
            MessageToast.show("Configuration dialog would open here");
        },

        onNavBack: function () {
            const oHistory = History.getInstance();
            const sPreviousHash = oHistory.getPreviousHash();

            if (sPreviousHash !== undefined) {
                window.history.go(-1);
            } else {
                this.getOwnerComponent().getRouter().navTo("home");
            }
        },

        onExit: function () {
            // Clean up refresh interval
            if (this._refreshInterval) {
                clearInterval(this._refreshInterval);
                this._refreshInterval = null;
            }
        }
    });
});