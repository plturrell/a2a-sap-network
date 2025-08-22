sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/ui/core/Fragment",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/model/json/JSONModel"
], function (ControllerExtension, Fragment, MessageBox, MessageToast, JSONModel) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent7.ext.controller.ListReportExt", {
        
        override: {
            onInit: function () {
                this._extensionAPI = this.base.getExtensionAPI();
                this._startRealtimeUpdates();
            }
        },

        onCreateManagementTask: function() {
            var oView = this.base.getView();
            
            if (!this._oCreateDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent7.ext.fragment.CreateManagementTask",
                    controller: this
                }).then(function(oDialog) {
                    this._oCreateDialog = oDialog;
                    oView.addDependent(this._oCreateDialog);
                    
                    var oModel = new JSONModel({
                        taskName: "",
                        description: "",
                        managedAgent: "",
                        operationType: "",
                        priority: "MEDIUM",
                        scheduledTime: null,
                        parameters: {},
                        notifyOnCompletion: true
                    });
                    this._oCreateDialog.setModel(oModel, "create");
                    this._oCreateDialog.open();
                    this._loadAvailableAgents();
                }.bind(this));
            } else {
                this._oCreateDialog.open();
                this._loadAvailableAgents();
            }
        },

        _loadAvailableAgents: function() {
            jQuery.ajax({
                url: "/a2a/agent7/v1/registered-agents",
                type: "GET",
                success: function(data) {
                    var oModel = this._oCreateDialog.getModel("create");
                    var oData = oModel.getData();
                    oData.availableAgents = data.agents;
                    oData.availableOperations = data.operations;
                    oModel.setData(oData);
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to load available agents: " + xhr.responseText);
                }.bind(this)
            });
        },

        onAgentDashboard: function() {
            var oView = this.base.getView();
            
            if (!this._oDashboard) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent7.ext.fragment.AgentDashboard",
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
                url: "/a2a/agent7/v1/dashboard",
                type: "GET",
                success: function(data) {
                    this._oDashboard.setBusy(false);
                    
                    var oDashboardModel = new JSONModel({
                        summary: data.summary,
                        agentHealth: data.agentHealth,
                        performance: data.performance,
                        operations: data.operations,
                        alerts: data.alerts,
                        trends: data.trends
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
            this._createAgentHealthChart(data.agentHealth);
            this._createPerformanceChart(data.performance);
            this._createOperationsChart(data.operations);
        },

        _createAgentHealthChart: function(healthData) {
            var oVizFrame = this._oDashboard.byId("agentHealthChart");
            if (!oVizFrame || !healthData) return;
            
            var aChartData = healthData.map(function(agent) {
                return {
                    Agent: agent.name,
                    Health: agent.healthScore,
                    Uptime: agent.uptime,
                    ResponseTime: agent.responseTime
                };
            });
            
            var oChartModel = new sap.ui.model.json.JSONModel({
                healthData: aChartData
            });
            oVizFrame.setModel(oChartModel);
            
            oVizFrame.setVizProperties({
                categoryAxis: {
                    title: { text: "Agents" }
                },
                valueAxis: {
                    title: { text: "Health Score" }
                },
                title: {
                    text: "Agent Health Overview"
                }
            });
        },

        onRegisterAgent: function() {
            var oView = this.base.getView();
            
            if (!this._oRegisterDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent7.ext.fragment.RegisterAgent",
                    controller: this
                }).then(function(oDialog) {
                    this._oRegisterDialog = oDialog;
                    oView.addDependent(this._oRegisterDialog);
                    
                    var oModel = new JSONModel({
                        agentName: "",
                        agentType: "",
                        version: "",
                        endpoint: "",
                        port: 8000,
                        capabilities: [],
                        dependencies: [],
                        configuration: {},
                        autoStart: true
                    });
                    this._oRegisterDialog.setModel(oModel, "register");
                    this._oRegisterDialog.open();
                    this._loadAgentTypes();
                }.bind(this));
            } else {
                this._oRegisterDialog.open();
                this._loadAgentTypes();
            }
        },

        _loadAgentTypes: function() {
            jQuery.ajax({
                url: "/a2a/agent7/v1/agent-types",
                type: "GET",
                success: function(data) {
                    var oModel = this._oRegisterDialog.getModel("register");
                    var oData = oModel.getData();
                    oData.availableTypes = data.types;
                    oData.availableCapabilities = data.capabilities;
                    oModel.setData(oData);
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to load agent types: " + xhr.responseText);
                }.bind(this)
            });
        },

        onHealthMonitor: function() {
            var oView = this.base.getView();
            
            if (!this._oHealthMonitor) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent7.ext.fragment.HealthMonitor",
                    controller: this
                }).then(function(oDialog) {
                    this._oHealthMonitor = oDialog;
                    oView.addDependent(this._oHealthMonitor);
                    this._oHealthMonitor.open();
                    this._loadHealthData();
                    this._startHealthMonitoring();
                }.bind(this));
            } else {
                this._oHealthMonitor.open();
                this._loadHealthData();
            }
        },

        _loadHealthData: function() {
            jQuery.ajax({
                url: "/a2a/agent7/v1/health-status",
                type: "GET",
                success: function(data) {
                    var oModel = new JSONModel({
                        healthChecks: data.healthChecks,
                        systemHealth: data.systemHealth,
                        alerts: data.alerts,
                        recommendations: data.recommendations
                    });
                    this._oHealthMonitor.setModel(oModel, "health");
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to load health data: " + xhr.responseText);
                }.bind(this)
            });
        },

        _startHealthMonitoring: function() {
            this._healthEventSource = new EventSource("/a2a/agent7/v1/health-stream");
            
            this._healthEventSource.onmessage = function(event) {
                var data = JSON.parse(event.data);
                this._updateHealthDisplay(data);
            }.bind(this);
            
            this._healthEventSource.onerror = function() {
                MessageToast.show("Health monitoring connection lost");
            }.bind(this);
        },

        _updateHealthDisplay: function(healthData) {
            var oModel = this._oHealthMonitor.getModel("health");
            if (!oModel) return;
            
            var oData = oModel.getData();
            
            if (healthData.type === "health_update") {
                var agentIndex = oData.healthChecks.findIndex(function(agent) {
                    return agent.agentId === healthData.agentId;
                });
                
                if (agentIndex >= 0) {
                    oData.healthChecks[agentIndex] = healthData.health;
                    oModel.setData(oData);
                }
            } else if (healthData.type === "alert") {
                oData.alerts.unshift(healthData.alert);
                if (oData.alerts.length > 50) {
                    oData.alerts.pop();
                }
                oModel.setData(oData);
                
                if (healthData.alert.severity === "CRITICAL") {
                    MessageToast.show("Critical Alert: " + healthData.alert.message);
                }
            }
        },

        onPerformanceAnalyzer: function() {
            var oView = this.base.getView();
            
            if (!this._oPerformanceDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent7.ext.fragment.PerformanceAnalyzer",
                    controller: this
                }).then(function(oDialog) {
                    this._oPerformanceDialog = oDialog;
                    oView.addDependent(this._oPerformanceDialog);
                    this._oPerformanceDialog.open();
                    this._loadPerformanceData();
                }.bind(this));
            } else {
                this._oPerformanceDialog.open();
                this._loadPerformanceData();
            }
        },

        _loadPerformanceData: function() {
            jQuery.ajax({
                url: "/a2a/agent7/v1/performance-metrics",
                type: "GET",
                success: function(data) {
                    var oModel = new JSONModel({
                        metrics: data.metrics,
                        trends: data.trends,
                        benchmarks: data.benchmarks,
                        recommendations: data.recommendations
                    });
                    this._oPerformanceDialog.setModel(oModel, "performance");
                    this._createPerformanceCharts(data);
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to load performance data: " + xhr.responseText);
                }.bind(this)
            });
        },

        _createPerformanceCharts: function(data) {
            var oResponseTimeChart = this._oPerformanceDialog.byId("responseTimeChart");
            var oThroughputChart = this._oPerformanceDialog.byId("throughputChart");
            
            if (oResponseTimeChart && data.trends) {
                var aResponseData = data.trends.map(function(trend) {
                    return {
                        Time: trend.timestamp,
                        ResponseTime: trend.averageResponseTime
                    };
                });
                
                var oResponseModel = new sap.ui.model.json.JSONModel({
                    responseData: aResponseData
                });
                oResponseTimeChart.setModel(oResponseModel);
            }
            
            if (oThroughputChart && data.trends) {
                var aThroughputData = data.trends.map(function(trend) {
                    return {
                        Time: trend.timestamp,
                        Throughput: trend.throughput
                    };
                });
                
                var oThroughputModel = new sap.ui.model.json.JSONModel({
                    throughputData: aThroughputData
                });
                oThroughputChart.setModel(oThroughputModel);
            }
        },

        onAgentCoordinator: function() {
            var oView = this.base.getView();
            
            if (!this._oCoordinatorDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent7.ext.fragment.AgentCoordinator",
                    controller: this
                }).then(function(oDialog) {
                    this._oCoordinatorDialog = oDialog;
                    oView.addDependent(this._oCoordinatorDialog);
                    this._oCoordinatorDialog.open();
                    this._loadCoordinationData();
                }.bind(this));
            } else {
                this._oCoordinatorDialog.open();
                this._loadCoordinationData();
            }
        },

        _loadCoordinationData: function() {
            jQuery.ajax({
                url: "/a2a/agent7/v1/coordination-status",
                type: "GET",
                success: function(data) {
                    var oModel = new JSONModel({
                        activeConnections: data.activeConnections,
                        workflowStatus: data.workflowStatus,
                        loadBalancing: data.loadBalancing,
                        failoverStatus: data.failoverStatus
                    });
                    this._oCoordinatorDialog.setModel(oModel, "coordination");
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to load coordination data: " + xhr.responseText);
                }.bind(this)
            });
        },

        onBulkOperations: function() {
            var oTable = this._extensionAPI.getTable();
            var aSelectedContexts = oTable.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageBox.warning("Please select at least one agent for bulk operations.");
                return;
            }
            
            var oView = this.base.getView();
            
            if (!this._oBulkDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent7.ext.fragment.BulkOperations",
                    controller: this
                }).then(function(oDialog) {
                    this._oBulkDialog = oDialog;
                    oView.addDependent(this._oBulkDialog);
                    
                    var aAgentIds = aSelectedContexts.map(function(oContext) {
                        return oContext.getProperty("managedAgent");
                    });
                    
                    var oModel = new JSONModel({
                        selectedAgents: aAgentIds,
                        operation: "",
                        parameters: {},
                        executeInParallel: true,
                        rollbackOnFailure: true
                    });
                    this._oBulkDialog.setModel(oModel, "bulk");
                    this._oBulkDialog.open();
                }.bind(this));
            } else {
                this._oBulkDialog.open();
            }
        },

        _startRealtimeUpdates: function() {
            this._realtimeEventSource = new EventSource("/a2a/agent7/v1/realtime-updates");
            
            this._realtimeEventSource.onmessage = function(event) {
                var data = JSON.parse(event.data);
                
                if (data.type === "agent_status_change") {
                    this._extensionAPI.refresh();
                    MessageToast.show("Agent " + data.agentName + " status changed to " + data.status);
                } else if (data.type === "performance_alert") {
                    MessageToast.show("Performance Alert: " + data.message);
                }
            }.bind(this);
            
            this._realtimeEventSource.onerror = function() {
                MessageToast.show("Real-time updates disconnected");
            }.bind(this);
        },

        onConfirmCreateTask: function() {
            var oModel = this._oCreateDialog.getModel("create");
            var oData = oModel.getData();
            
            if (!oData.taskName || !oData.managedAgent || !oData.operationType) {
                MessageBox.error("Please fill all required fields");
                return;
            }
            
            this._oCreateDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent7/v1/management-tasks",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify(oData),
                success: function(data) {
                    this._oCreateDialog.setBusy(false);
                    this._oCreateDialog.close();
                    MessageToast.show("Management task created successfully");
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    this._oCreateDialog.setBusy(false);
                    MessageBox.error("Failed to create task: " + xhr.responseText);
                }.bind(this)
            });
        },

        onConfirmRegisterAgent: function() {
            var oModel = this._oRegisterDialog.getModel("register");
            var oData = oModel.getData();
            
            if (!oData.agentName || !oData.agentType || !oData.endpoint) {
                MessageBox.error("Please fill all required fields");
                return;
            }
            
            this._oRegisterDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent7/v1/register-agent",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify(oData),
                success: function(data) {
                    this._oRegisterDialog.setBusy(false);
                    this._oRegisterDialog.close();
                    MessageBox.success(
                        "Agent registered successfully!\\n" +
                        "Agent ID: " + data.agentId + "\\n" +
                        "Registration Block: " + data.blockNumber
                    );
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    this._oRegisterDialog.setBusy(false);
                    MessageBox.error("Agent registration failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        onCancelCreateTask: function() {
            this._oCreateDialog.close();
        },

        onCancelRegisterAgent: function() {
            this._oRegisterDialog.close();
        }
    });
});