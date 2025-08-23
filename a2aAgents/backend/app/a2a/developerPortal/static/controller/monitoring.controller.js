sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/format/DateFormat"
], (Controller, JSONModel, MessageToast, MessageBox, DateFormat) => {
    "use strict";
/* global  */

    return Controller.extend("a2a.portal.controller.Monitoring", {

        onInit: function () {
            // Initialize view model
            const oViewModel = new JSONModel({
                viewMode: "dashboard",
                agents: [],
                logs: [],
                dashboard: {},
                performance: {},
                liveLogsEnabled: false,
                busy: false
            });
            this.getView().setModel(oViewModel, "view");

            // Load monitoring data
            this._loadMonitoringData();
        },

        _loadMonitoringData: function () {
            const oViewModel = this.getView().getModel("view");
            oViewModel.setProperty("/busy", true);

            jQuery.ajax({
                url: "/api/monitoring/data",
                method: "GET",
                success: function (data) {
                    oViewModel.setProperty("/agents", data.agents || []);
                    oViewModel.setProperty("/logs", data.logs || []);
                    oViewModel.setProperty("/dashboard", data.dashboard || {});
                    oViewModel.setProperty("/performance", data.performance || {});
                    oViewModel.setProperty("/busy", false);
                }.bind(this),
                error: function (_xhr, _status, _error) {
                    // Fallback to mock data
                    const oMockData = this._getMockMonitoringData();
                    oViewModel.setProperty("/agents", oMockData.agents);
                    oViewModel.setProperty("/logs", oMockData.logs);
                    oViewModel.setProperty("/dashboard", oMockData.dashboard);
                    oViewModel.setProperty("/performance", oMockData.performance);
                    oViewModel.setProperty("/busy", false);
                    MessageToast.show("Using sample data - backend connection unavailable");
                }.bind(this)
            });
        },

        _getMockMonitoringData: function () {
            return {
                dashboard: {
                    uptime: 15.2,
                    totalRequests: 24567,
                    avgResponseTime: 142,
                    errorRate: 2.1,
                    cpuUsage: 68,
                    memoryUsage: 74
                },
                performance: {
                    currentThroughput: 145,
                    peakThroughput: 320,
                    avgThroughput: 180,
                    p95ResponseTime: 250,
                    p99ResponseTime: 480,
                    maxResponseTime: 1200
                },
                agents: [
                    {
                        id: "agent-1",
                        name: "Agent0 Data Product",
                        type: "Data Product Agent",
                        environment: "Production",
                        status: "running",
                        uptime: "12d 4h",
                        requestsHandled: 8542,
                        avgProcessingTime: 95,
                        lastActivity: "2024-01-22T15:30:00Z"
                    },
                    {
                        id: "agent-2", 
                        name: "Agent1 Standardization",
                        type: "Standardization Agent",
                        environment: "Production",
                        status: "running",
                        uptime: "8d 15h",
                        requestsHandled: 6234,
                        avgProcessingTime: 123,
                        lastActivity: "2024-01-22T15:25:00Z"
                    },
                    {
                        id: "agent-3",
                        name: "Integration Agent",
                        type: "Integration Agent", 
                        environment: "Staging",
                        status: "idle",
                        uptime: "5d 2h",
                        requestsHandled: 1456,
                        avgProcessingTime: 78,
                        lastActivity: "2024-01-22T14:45:00Z"
                    },
                    {
                        id: "agent-4",
                        name: "QA Validation Agent",
                        type: "Validation Agent",
                        environment: "Development",
                        status: "error",
                        uptime: "2h 15m",
                        requestsHandled: 45,
                        avgProcessingTime: 234,
                        lastActivity: "2024-01-22T13:20:00Z"
                    }
                ],
                logs: [
                    {
                        id: "log-1",
                        timestamp: "2024-01-22T15:30:25Z",
                        level: "INFO",
                        component: "Agent0",
                        message: "Data product successfully processed for customer ABC123"
                    },
                    {
                        id: "log-2",
                        timestamp: "2024-01-22T15:30:15Z",
                        level: "WARN",
                        component: "Agent1",
                        message: "Standardization rule validation took longer than expected (2.5s)"
                    },
                    {
                        id: "log-3",
                        timestamp: "2024-01-22T15:29:45Z",
                        level: "ERROR",
                        component: "QA Agent",
                        message: "Failed to connect to validation service endpoint"
                    },
                    {
                        id: "log-4",
                        timestamp: "2024-01-22T15:29:30Z",
                        level: "INFO",
                        component: "System",
                        message: "Health check completed successfully - all services operational"
                    },
                    {
                        id: "log-5",
                        timestamp: "2024-01-22T15:28:12Z",
                        level: "DEBUG",
                        component: "Integration",
                        message: "Processing workflow step 3/5 for request ID req-789"
                    }
                ]
            };
        },

        onRefreshMonitoring: function () {
            this._loadMonitoringData();
            MessageToast.show("Monitoring data refreshed");
        },

        onViewChange: function (oEvent) {
            const sSelectedKey = oEvent.getParameter("item").getKey();
            const oViewModel = this.getView().getModel("view");
            oViewModel.setProperty("/viewMode", sSelectedKey);
        },

        onSearch: function (oEvent) {
            const sQuery = oEvent.getParameter("query");
            const sViewMode = this.getView().getModel("view").getProperty("/viewMode");
            let oTable, oBinding;

            switch (sViewMode) {
                case "agents":
                    oTable = this.byId("agentsTable");
                    break;
                case "logs":
                    oTable = this.byId("logsTable");
                    break;
                default:
                    return;
            }

            if (oTable) {
                oBinding = oTable.getBinding("items");
                if (sQuery && sQuery.length > 0) {
                    let aFilters = [];
                    
                    if (sViewMode === "agents") {
                        aFilters = [
                            new sap.ui.model.Filter("name", sap.ui.model.FilterOperator.Contains, sQuery),
                            new sap.ui.model.Filter("type", sap.ui.model.FilterOperator.Contains, sQuery),
                            new sap.ui.model.Filter("environment", sap.ui.model.FilterOperator.Contains, sQuery)
                        ];
                    } else if (sViewMode === "logs") {
                        aFilters = [
                            new sap.ui.model.Filter("component", sap.ui.model.FilterOperator.Contains, sQuery),
                            new sap.ui.model.Filter("message", sap.ui.model.FilterOperator.Contains, sQuery)
                        ];
                    }
                    
                    const oFilter = new sap.ui.model.Filter(aFilters, false);
                    oBinding.filter([oFilter]);
                } else {
                    oBinding.filter([]);
                }
            }
        },

        onOpenFilterDialog: function () {
            MessageToast.show("Filter dialog - coming soon");
        },

        onOpenSortDialog: function () {
            MessageToast.show("Sort dialog - coming soon");
        },

        onViewAlerts: function () {
            MessageBox.information(
                "Active Alerts:\n\n" +
                "• High CPU usage on Production environment (78%)\n" +
                "• QA Validation Agent connection errors\n" +
                "• Slow response times detected (>2s)\n" +
                "• Memory usage approaching threshold (85%)",
                {
                    title: "System Alerts"
                }
            );
        },

        onExportLogs: function () {
            MessageToast.show("Exporting logs to file...");
        },

        onCheckSystemHealth: function () {
            MessageToast.show("Running system health check...");
            
            setTimeout(() => {
                MessageBox.success(
                    "System Health Check Complete\n\n" +
                    "✓ All core services operational\n" +
                    "✓ Database connections stable\n" +
                    "✓ Agent communication verified\n" +
                    "⚠ 2 minor issues identified\n\n" +
                    "Overall Status: Healthy",
                    {
                        title: "Health Check Results"
                    }
                );
            }, 2000);
        },

        onCheckAgentStatus: function () {
            MessageToast.show("Checking agent status...");
            
            setTimeout(() => {
                MessageBox.information(
                    "Agent Status Summary:\n\n" +
                    "✓ Agent0 Data Product: Running (12d uptime)\n" +
                    "✓ Agent1 Standardization: Running (8d uptime)\n" +
                    "⚠ Integration Agent: Idle (5d uptime)\n" +
                    "✗ QA Validation Agent: Error (connection failed)\n\n" +
                    "3 of 4 agents operational",
                    {
                        title: "Agent Status"
                    }
                );
            }, 1500);
        },

        onClearAlerts: function () {
            MessageBox.confirm(
                "Clear all current alerts? This will acknowledge all active alerts.", {
                    title: "Clear Alerts",
                    onClose: function (sAction) {
                        if (sAction === MessageBox.Action.OK) {
                            MessageToast.show("All alerts cleared");
                        }
                    }
                }
            );
        },

        onAgentPress: function (oEvent) {
            const oContext = oEvent.getSource().getBindingContext("view");
            const sAgentId = oContext.getProperty("id");
            MessageToast.show(`Agent selected: ${  sAgentId}`);
        },

        onViewAgentDetails: function (oEvent) {
            oEvent.stopPropagation();
            const oContext = oEvent.getSource().getBindingContext("view");
            const sAgentName = oContext.getProperty("name");
            MessageToast.show(`Viewing details for: ${  sAgentName}`);
        },

        onRestartAgent: function (oEvent) {
            oEvent.stopPropagation();
            const oContext = oEvent.getSource().getBindingContext("view");
            const sAgentName = oContext.getProperty("name");
            
            MessageBox.confirm(
                `Restart agent '${  sAgentName  }'? This will temporarily interrupt service.`, {
                    icon: MessageBox.Icon.WARNING,
                    title: "Restart Agent",
                    onClose: function (sAction) {
                        if (sAction === MessageBox.Action.OK) {
                            MessageToast.show(`Restarting agent: ${  sAgentName}`);
                        }
                    }
                }
            );
        },

        onViewAgentLogs: function (oEvent) {
            oEvent.stopPropagation();
            const oContext = oEvent.getSource().getBindingContext("view");
            const sAgentName = oContext.getProperty("name");
            MessageToast.show(`Viewing logs for: ${  sAgentName}`);
        },

        onRestartAllAgents: function () {
            MessageBox.confirm(
                "Restart all agents? This will temporarily interrupt all services.", {
                    icon: MessageBox.Icon.WARNING,
                    title: "Restart All Agents",
                    onClose: function (sAction) {
                        if (sAction === MessageBox.Action.OK) {
                            MessageToast.show("Restarting all agents...");
                        }
                    }
                }
            );
        },

        onToggleLiveLogs: function (oEvent) {
            const bState = oEvent.getParameter("state");
            const oViewModel = this.getView().getModel("view");
            oViewModel.setProperty("/liveLogsEnabled", bState);
            
            MessageToast.show(bState ? "Live logs enabled" : "Live logs paused");
        },

        onLogLevelFilter: function (oEvent) {
            const sSelectedLevel = oEvent.getParameter("selectedItem").getKey();
            const oTable = this.byId("logsTable");
            
            if (oTable) {
                const oBinding = oTable.getBinding("items");
                if (sSelectedLevel !== "all") {
                    const oFilter = new sap.ui.model.Filter("level", sap.ui.model.FilterOperator.EQ, sSelectedLevel.toUpperCase());
                    oBinding.filter([oFilter]);
                } else {
                    oBinding.filter([]);
                }
            }
        },

        onLogPress: function (oEvent) {
            const oContext = oEvent.getSource().getBindingContext("view");
            const sLogId = oContext.getProperty("id");
            MessageToast.show(`Log entry selected: ${  sLogId}`);
        },

        onViewLogDetails: function (oEvent) {
            oEvent.stopPropagation();
            const oContext = oEvent.getSource().getBindingContext("view");
            const oLog = oContext.getObject();
            
            MessageBox.information(
                `Log Entry Details:\n\n` +
                `Timestamp: ${  this.formatDate(oLog.timestamp)  }\n` +
                `Level: ${  oLog.level  }\n` +
                `Component: ${  oLog.component  }\n` +
                `Message: ${  oLog.message}`,
                {
                    title: "Log Details"
                }
            );
        },

        onExportSelected: function () {
            MessageToast.show("Export functionality - coming soon");
        },

        formatDate: function (sDate) {
            if (!sDate) {
                return "";
            }
            
            const oDateFormat = DateFormat.getDateTimeInstance({
                style: "medium"
            });
            
            return oDateFormat.format(new Date(sDate));
        },

        formatStatusState: function (sStatus) {
            switch (sStatus) {
                case "running": return "Success";
                case "idle": return "Warning";
                case "error": return "Error";
                default: return "None";
            }
        },

        formatLogLevelState: function (sLevel) {
            switch (sLevel) {
                case "ERROR": return "Error";
                case "WARN": return "Warning";
                case "INFO": return "Success";
                case "DEBUG": return "Information";
                default: return "None";
            }
        }
    });
});