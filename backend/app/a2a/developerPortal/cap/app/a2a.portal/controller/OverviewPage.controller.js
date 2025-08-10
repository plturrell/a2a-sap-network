sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast",
    "sap/m/MessageBox"
], function (Controller, JSONModel, MessageToast, MessageBox) {
    "use strict";

    return Controller.extend("com.sap.a2a.developerportal.controller.OverviewPage", {

        onInit: function () {
            this.oRouter = this.getOwnerComponent().getRouter();
            
            // Initialize dashboard model
            this._initializeDashboardModel();
            
            // Load dashboard data
            this._loadDashboardData();
            
            // Set up periodic refresh
            this._setupPeriodicRefresh();
        },

        _initializeDashboardModel: function () {
            var oDashboardModel = new JSONModel({
                kpis: {
                    active_projects: 0,
                    total_agents: 0,
                    deployments_this_month: 0,
                    success_rate: "0%"
                },
                recent_projects: [],
                system_health: {
                    cpu_usage: 0,
                    memory_usage: 0,
                    active_connections: 0
                },
                recent_activity: [],
                notifications: []
            });
            
            this.getView().setModel(oDashboardModel);
        },

        _loadDashboardData: function () {
            this._loadKPIs();
            this._loadRecentProjects();
            this._loadSystemHealth();
            this._loadRecentActivity();
            this._loadNotifications();
        },

        _loadKPIs: function () {
            var that = this;
            
            jQuery.ajax({
                url: "/api/v2/dashboard/kpis",
                method: "GET",
                success: function (oData) {
                    var oModel = that.getView().getModel();
                    oModel.setProperty("/kpis", oData);
                },
                error: function () {
                    // Fallback to mock data
                    var oModel = that.getView().getModel();
                    oModel.setProperty("/kpis", {
                        active_projects: 12,
                        total_agents: 45,
                        deployments_this_month: 8,
                        success_rate: "98.5%"
                    });
                }
            });
        },

        _loadRecentProjects: function () {
            var that = this;
            
            jQuery.ajax({
                url: "/api/v2/projects?limit=5&sort=updated_at&order=desc",
                method: "GET",
                success: function (oData) {
                    var oModel = that.getView().getModel();
                    oModel.setProperty("/recent_projects", oData.projects || []);
                },
                error: function () {
                    // Fallback to mock data
                    var oModel = that.getView().getModel();
                    oModel.setProperty("/recent_projects", [
                        {
                            project_id: "proj1",
                            name: "Customer Data Pipeline",
                            description: "Processes customer data from multiple sources",
                            status: "active",
                            type: "Data Processing",
                            agents_count: 5,
                            updated_at: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString()
                        },
                        {
                            project_id: "proj2",
                            name: "Order Management System",
                            description: "Automates order processing workflows",
                            status: "deployed",
                            type: "Workflow Automation",
                            agents_count: 8,
                            updated_at: new Date(Date.now() - 4 * 60 * 60 * 1000).toISOString()
                        }
                    ]);
                }
            });
        },

        _loadSystemHealth: function () {
            var that = this;
            
            jQuery.ajax({
                url: "/api/v2/system/health",
                method: "GET",
                success: function (oData) {
                    var oModel = that.getView().getModel();
                    oModel.setProperty("/system_health", oData);
                },
                error: function () {
                    // Fallback to mock data
                    var oModel = that.getView().getModel();
                    oModel.setProperty("/system_health", {
                        cpu_usage: 35,
                        memory_usage: 62,
                        active_connections: 24
                    });
                }
            });
        },

        _loadRecentActivity: function () {
            var that = this;
            
            jQuery.ajax({
                url: "/api/v2/activity?limit=10",
                method: "GET",
                success: function (oData) {
                    var oModel = that.getView().getModel();
                    oModel.setProperty("/recent_activity", oData.activities || []);
                },
                error: function () {
                    // Fallback to mock data
                    var oModel = that.getView().getModel();
                    oModel.setProperty("/recent_activity", [
                        {
                            id: "act1",
                            user: "System",
                            description: "Project 'Customer Data Pipeline' deployed successfully",
                            timestamp: new Date(Date.now() - 30 * 60 * 1000).toISOString(),
                            type: "deployment"
                        },
                        {
                            id: "act2",
                            user: "Developer",
                            description: "New agent 'Data Validator' created",
                            timestamp: new Date(Date.now() - 60 * 60 * 1000).toISOString(),
                            type: "creation"
                        },
                        {
                            id: "act3",
                            user: "System",
                            description: "Workflow 'Order Processing' executed successfully",
                            timestamp: new Date(Date.now() - 90 * 60 * 1000).toISOString(),
                            type: "execution"
                        }
                    ]);
                }
            });
        },

        _loadNotifications: function () {
            var that = this;
            
            jQuery.ajax({
                url: "/api/v2/notifications?unread=true",
                method: "GET",
                success: function (oData) {
                    var oModel = that.getView().getModel();
                    oModel.setProperty("/notifications", oData.notifications || []);
                },
                error: function () {
                    // Fallback to mock data
                    var oModel = that.getView().getModel();
                    oModel.setProperty("/notifications", [
                        {
                            id: "notif1",
                            title: "Deployment Completed",
                            description: "Project 'Customer Data Pipeline' has been successfully deployed to production",
                            timestamp: new Date(Date.now() - 15 * 60 * 1000).toISOString(),
                            priority: "Medium"
                        },
                        {
                            id: "notif2",
                            title: "System Alert",
                            description: "High memory usage detected on server node-02",
                            timestamp: new Date(Date.now() - 45 * 60 * 1000).toISOString(),
                            priority: "High"
                        }
                    ]);
                }
            });
        },

        _setupPeriodicRefresh: function () {
            // Refresh dashboard data every 5 minutes
            setInterval(function () {
                this._loadSystemHealth();
                this._loadNotifications();
            }.bind(this), 5 * 60 * 1000);
        },

        // Header Actions
        onRefresh: function () {
            this._loadDashboardData();
            MessageToast.show("Dashboard refreshed");
        },

        onCustomizeDashboard: function () {
            MessageToast.show("Dashboard customization - Coming soon");
        },

        onQuickStart: function () {
            MessageBox.information(
                "Welcome to A2A Agents!\n\n" +
                "1. Create a new project\n" +
                "2. Design your agents\n" +
                "3. Configure workflows\n" +
                "4. Test and validate\n" +
                "5. Deploy to production",
                {
                    title: "Quick Start Guide"
                }
            );
        },

        // Welcome Card Actions
        onCreateProject: function () {
            this.oRouter.navTo("projectCreate");
        },

        onImportProject: function () {
            MessageToast.show("Import project functionality");
        },

        onViewDocs: function () {
            window.open("https://help.sap.com/", "_blank");
        },

        // KPI Actions
        onProjectsKPIPress: function () {
            this.oRouter.navTo("projects");
        },

        onAgentsKPIPress: function () {
            this.oRouter.navTo("agentsOverview");
        },

        onDeploymentsKPIPress: function () {
            this.oRouter.navTo("deploymentsOverview");
        },

        onSuccessRateKPIPress: function () {
            this.oRouter.navTo("performanceDashboard");
        },

        onDrillDownProjects: function () {
            this.oRouter.navTo("projectsSmart");
        },

        // Card Actions
        onViewAllProjects: function () {
            this.oRouter.navTo("projects");
        },

        onProjectPress: function (oEvent) {
            var oBindingContext = oEvent.getSource().getBindingContext();
            var sProjectId = oBindingContext.getProperty("project_id");
            
            this.oRouter.navTo("projectDetail", {
                projectId: sProjectId
            });
        },

        onViewSystemHealth: function () {
            this.oRouter.navTo("systemHealth");
        },

        onViewAllActivity: function () {
            this.oRouter.navTo("activityLog");
        },

        onActivityPress: function (oEvent) {
            var oBindingContext = oEvent.getSource().getBindingContext();
            var oActivity = oBindingContext.getObject();
            
            MessageToast.show("Activity: " + oActivity.description);
        },

        // Quick Actions
        onQuickAgentBuilder: function () {
            this.oRouter.navTo("agentBuilder");
        },

        onQuickBPMNDesigner: function () {
            this.oRouter.navTo("bpmnDesigner");
        },

        onQuickTesting: function () {
            this.oRouter.navTo("testingFramework");
        },

        onQuickDeploy: function () {
            this.oRouter.navTo("deploymentPipeline");
        },

        onViewPerformanceDetails: function () {
            this.oRouter.navTo("performanceDashboard");
        },

        // Notifications
        onMarkAllRead: function () {
            var that = this;
            
            jQuery.ajax({
                url: "/api/v2/notifications/mark-all-read",
                method: "POST",
                success: function () {
                    MessageToast.show("All notifications marked as read");
                    that._loadNotifications();
                },
                error: function () {
                    MessageToast.show("Failed to mark notifications as read");
                }
            });
        },

        onNotificationSettings: function () {
            this.oRouter.navTo("notificationSettings");
        },

        onCloseNotification: function (oEvent) {
            var oNotification = oEvent.getSource();
            var sNotificationId = oNotification.getCustomData()[0].getValue();
            
            this._markNotificationRead(sNotificationId);
        },

        onNotificationPress: function (oEvent) {
            var oBindingContext = oEvent.getSource().getBindingContext();
            var oNotification = oBindingContext.getObject();
            
            // Navigate based on notification type or show details
            MessageToast.show("Notification: " + oNotification.title);
        },

        _markNotificationRead: function (sNotificationId) {
            var that = this;
            
            jQuery.ajax({
                url: "/api/v2/notifications/" + sNotificationId + "/read",
                method: "POST",
                success: function () {
                    that._loadNotifications();
                },
                error: function () {
                    MessageToast.show("Failed to mark notification as read");
                }
            });
        },

        // Formatters
        formatDate: function (sDate) {
            if (!sDate) return "";
            
            var oDate = new Date(sDate);
            return oDate.toLocaleDateString();
        },

        formatRelativeTime: function (sDate) {
            if (!sDate) return "";
            
            var oDate = new Date(sDate);
            var oNow = new Date();
            var iDiff = oNow.getTime() - oDate.getTime();
            var iMinutes = Math.floor(iDiff / (1000 * 60));
            var iHours = Math.floor(iMinutes / 60);
            var iDays = Math.floor(iHours / 24);
            
            if (iMinutes < 1) {
                return "Just now";
            } else if (iMinutes < 60) {
                return iMinutes + " minutes ago";
            } else if (iHours < 24) {
                return iHours + " hours ago";
            } else if (iDays === 1) {
                return "Yesterday";
            } else if (iDays < 7) {
                return iDays + " days ago";
            } else {
                return oDate.toLocaleDateString();
            }
        },

        formatStatusState: function (sStatus) {
            switch (sStatus) {
                case "active":
                    return "Success";
                case "deployed":
                    return "Success";
                case "inactive":
                    return "Warning";
                case "error":
                    return "Error";
                default:
                    return "None";
            }
        },

        formatAgentsState: function (iCount) {
            if (iCount > 10) {
                return "Success";
            } else if (iCount > 5) {
                return "Warning";
            } else {
                return "None";
            }
        },

        formatHealthState: function (iValue) {
            if (iValue < 50) {
                return "Success";
            } else if (iValue < 80) {
                return "Warning";
            } else {
                return "Error";
            }
        },

        formatActivityIcon: function (sType) {
            switch (sType) {
                case "deployment":
                    return "sap-icon://cloud";
                case "creation":
                    return "sap-icon://add";
                case "execution":
                    return "sap-icon://play";
                case "configuration":
                    return "sap-icon://settings";
                default:
                    return "sap-icon://information";
            }
        }
    });
});
