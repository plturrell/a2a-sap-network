sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/core/Fragment",
    "sap/ui/model/Filter",
    "sap/ui/model/FilterOperator",
    "../BaseController",
    "../../utils/formatter"
], (Controller, JSONModel, MessageBox, MessageToast, Fragment, Filter, FilterOperator, BaseController, formatter) => {
    "use strict";

    return BaseController.extend("com.sap.a2a.a2aFiori.ext.deployment.DeploymentDashboard", {

        formatter,

        onInit() {
            // Initialize deployment model
            const oDeploymentModel = new JSONModel({
                activeDeployments: [],
                deploymentHistory: [],
                production: {
                    status: "Healthy",
                    state: "Success",
                    lastDeployment: "2 hours ago"
                },
                staging: {
                    status: "Deploying",
                    state: "Warning",
                    lastDeployment: "10 minutes ago"
                },
                systemHealth: {
                    score: 95,
                    state: "Success",
                    activeAgents: 16,
                    totalAgents: 18
                },
                selectedEnvironment: "all",
                deploymentTrend: [],
                environmentStats: []
            });

            this.getView().setModel(oDeploymentModel, "deploymentModel");

            // Set up real-time updates
            this._startRealTimeUpdates();

            // Load initial data
            this._loadDeploymentData();

            // Subscribe to deployment events
            this._subscribeToEvents();
        },

        onAfterRendering() {
            // Initialize charts
            this._initializeCharts();
        },

        onExit() {
            // Clean up
            if (this._updateInterval) {
                clearInterval(this._updateInterval);
            }
            if (this._eventSource) {
                this._eventSource.close();
            }
        },

        onNewDeployment() {
            if (!this._oNewDeploymentDialog) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "com.sap.a2a.a2aFiori.ext.deployment.NewDeploymentDialog",
                    controller: this
                }).then((oDialog) => {
                    this._oNewDeploymentDialog = oDialog;
                    this.getView().addDependent(oDialog);
                    oDialog.open();
                });
            } else {
                this._oNewDeploymentDialog.open();
            }
        },

        onRefresh() {
            this._loadDeploymentData();
            MessageToast.show("Deployment data refreshed");
        },

        onViewLogs(oEvent) {
            const oContext = oEvent.getSource().getBindingContext("deploymentModel");
            const deploymentId = oContext.getProperty("id");

            // Open log viewer
            if (!this._oLogViewerDialog) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "com.sap.a2a.a2aFiori.ext.deployment.LogViewerDialog",
                    controller: this
                }).then((oDialog) => {
                    this._oLogViewerDialog = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadDeploymentLogs(deploymentId);
                    oDialog.open();
                });
            } else {
                this._loadDeploymentLogs(deploymentId);
                this._oLogViewerDialog.open();
            }
        },

        onPauseDeployment(oEvent) {
            const oContext = oEvent.getSource().getBindingContext("deploymentModel");
            const deployment = oContext.getObject();

            MessageBox.confirm("Are you sure you want to pause this deployment?", {
                title: "Pause Deployment",
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        this._pauseDeployment(deployment.id);
                    }
                }.bind(this)
            });
        },

        onViewDetails(oEvent) {
            const oContext = oEvent.getSource().getBindingContext("deploymentModel");
            const deploymentId = oContext.getProperty("ID");

            // Navigate to detail view
            this.getRouter().navTo("deploymentDetail", {
                deploymentId
            });
        },

        onRollback(oEvent) {
            const oContext = oEvent.getSource().getBindingContext("deploymentModel");
            const deployment = oContext.getObject();

            MessageBox.confirm(
                `Are you sure you want to rollback ${deployment.appName} in ${deployment.environment}?`,
                {
                    title: "Confirm Rollback",
                    actions: [MessageBox.Action.YES, MessageBox.Action.NO],
                    emphasizedAction: MessageBox.Action.NO,
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.YES) {
                            this._performRollback(deployment.ID);
                        }
                    }.bind(this)
                }
            );
        },

        onSearchDeployments(oEvent) {
            const sQuery = oEvent.getParameter("query");
            const oTable = this.byId("deploymentHistoryTable");
            const oBinding = oTable.getBinding("items");

            if (sQuery) {
                const aFilters = [
                    new Filter("appName", FilterOperator.Contains, sQuery),
                    new Filter("version", FilterOperator.Contains, sQuery),
                    new Filter("deployedBy", FilterOperator.Contains, sQuery)
                ];
                oBinding.filter(new Filter(aFilters, false));
            } else {
                oBinding.filter([]);
            }
        },

        onStartDeployment() {
            const oModel = this.getView().getModel("deploymentModel");
            const oNewDeployment = oModel.getProperty("/newDeployment");

            // Validate input
            if (!oNewDeployment.appName || !oNewDeployment.environment || !oNewDeployment.version) {
                MessageBox.error("Please fill in all required fields");
                return;
            }

            // Call deployment service
            this._createDeployment(oNewDeployment);

            // Close dialog
            this._oNewDeploymentDialog.close();
        },

        onCancelDeployment() {
            this._oNewDeploymentDialog.close();
        },

        // Private methods
        _startRealTimeUpdates() {
            // Poll for updates every 5 seconds
            this._updateInterval = setInterval(() => {
                this._updateActiveDeployments();
                this._updateSystemHealth();
            }, 5000);
        },

        _subscribeToEvents() {
            // Set up SSE for real-time events
            if (typeof EventSource !== "undefined") {
                this._eventSource = new EventSource("/api/v1/deployment/events");

                this._eventSource.addEventListener("deploymentStarted", (e) => {
                    const data = JSON.parse(e.data);
                    this._handleDeploymentStarted(data);
                });

                this._eventSource.addEventListener("deploymentCompleted", (e) => {
                    const data = JSON.parse(e.data);
                    this._handleDeploymentCompleted(data);
                });

                this._eventSource.addEventListener("deploymentFailed", (e) => {
                    const data = JSON.parse(e.data);
                    this._handleDeploymentFailed(data);
                });
            }
        },

        _loadDeploymentData() {
            const oModel = this.getView().getModel("deploymentModel");

            // Load deployment history
            jQuery.ajax({
                url: "/api/v1/deployment/DeploymentSummary",
                method: "GET",
                success(data) {
                    oModel.setProperty("/deploymentHistory", data.value || []);
                },
                error(xhr) {
                    MessageBox.error("Failed to load deployment history");
                }
            });

            // Load active deployments
            this._updateActiveDeployments();

            // Load analytics data
            this._loadAnalytics();
        },

        _updateActiveDeployments() {
            jQuery.ajax({
                url: "/api/v1/deployment/getLiveDeploymentStatus",
                method: "GET",
                success: function(data) {
                    const oModel = this.getView().getModel("deploymentModel");
                    oModel.setProperty("/activeDeployments", data.activeDeployments || []);
                }.bind(this),
                error(xhr) {
                    console.error("Failed to update active deployments");
                }
            });
        },

        _updateSystemHealth() {
            jQuery.ajax({
                url: "/api/v1/deployment/getSystemHealth",
                method: "GET",
                success: function(data) {
                    const oModel = this.getView().getModel("deploymentModel");

                    // Update production status
                    oModel.setProperty("/production/status", data.production.status);
                    oModel.setProperty("/production/state", this._getStateFromStatus(data.production.status));

                    // Update staging status
                    oModel.setProperty("/staging/status", data.staging.status);
                    oModel.setProperty("/staging/state", this._getStateFromStatus(data.staging.status));

                    // Update system health
                    oModel.setProperty("/systemHealth/score", data.production.healthScore);
                    oModel.setProperty("/systemHealth/activeAgents", data.production.activeAgents);
                    oModel.setProperty("/systemHealth/totalAgents", data.production.totalAgents);
                    oModel.setProperty("/systemHealth/state",
                        data.production.healthScore >= 90 ? "Success" :
                            data.production.healthScore >= 70 ? "Warning" : "Error"
                    );
                }.bind(this),
                error(xhr) {
                    console.error("Failed to update system health");
                }
            });
        },

        _createDeployment(oDeploymentData) {
            jQuery.ajax({
                url: "/api/v1/deployment/createDeployment",
                method: "POST",
                contentType: "application/json",
                data: JSON.stringify(oDeploymentData),
                success: function(data) {
                    MessageToast.show("Deployment initiated successfully");
                    this._loadDeploymentData();
                }.bind(this),
                error(xhr) {
                    MessageBox.error(`Failed to create deployment: ${ xhr.responseText}`);
                }
            });
        },

        _performRollback(deploymentId) {
            jQuery.ajax({
                url: `/api/v1/deployment/Deployments(${deploymentId})/rollback`,
                method: "POST",
                success: function(data) {
                    MessageToast.show("Rollback initiated successfully");
                    this._loadDeploymentData();
                }.bind(this),
                error(xhr) {
                    MessageBox.error(`Failed to initiate rollback: ${ xhr.responseText}`);
                }
            });
        },

        _loadDeploymentLogs(deploymentId) {
            // Load logs for the deployment
            jQuery.ajax({
                url: `/api/v1/logs/stream?search=${deploymentId}&limit=1000`,
                method: "GET",
                success: function(data) {
                    const oLogModel = new JSONModel({
                        logs: data.logs || []
                    });
                    this._oLogViewerDialog.setModel(oLogModel, "logModel");
                }.bind(this),
                error(xhr) {
                    MessageBox.error("Failed to load deployment logs");
                }
            });
        },

        _loadAnalytics() {
            // Load deployment trend data
            jQuery.ajax({
                url: "/api/v1/deployment/analytics/trend",
                method: "GET",
                success: function(data) {
                    const oModel = this.getView().getModel("deploymentModel");
                    oModel.setProperty("/deploymentTrend", data || []);
                }.bind(this)
            });

            // Load environment statistics
            jQuery.ajax({
                url: "/api/v1/deployment/EnvironmentStatus",
                method: "GET",
                success: function(data) {
                    const oModel = this.getView().getModel("deploymentModel");
                    oModel.setProperty("/environmentStats", data.value || []);
                }.bind(this)
            });
        },

        _initializeCharts() {
            // Configure deployment trend chart
            const oTrendChart = this.byId("deploymentTrendChart");
            if (oTrendChart) {
                oTrendChart.setVizProperties({
                    plotArea: {
                        dataLabel: {
                            visible: true
                        }
                    },
                    valueAxis: {
                        title: {
                            visible: true
                        }
                    },
                    categoryAxis: {
                        title: {
                            visible: true
                        }
                    },
                    title: {
                        visible: true,
                        text: "Deployment Trends"
                    }
                });
            }

            // Configure environment chart
            const oEnvChart = this.byId("environmentChart");
            if (oEnvChart) {
                oEnvChart.setVizProperties({
                    plotArea: {
                        dataLabel: {
                            visible: true,
                            type: "percentage"
                        }
                    },
                    title: {
                        visible: true,
                        text: "Deployments by Environment"
                    }
                });
            }
        },

        _getStateFromStatus(status) {
            switch (status) {
            case "healthy":
            case "completed":
                return "Success";
            case "degraded":
            case "in_progress":
                return "Warning";
            case "critical":
            case "failed":
                return "Error";
            default:
                return "None";
            }
        },

        _handleDeploymentStarted(data) {
            MessageToast.show(`Deployment started: ${data.appName} v${data.version}`);
            this._loadDeploymentData();
        },

        _handleDeploymentCompleted(data) {
            MessageBox.success(`Deployment completed successfully: ${data.appName} in ${data.environment}`);
            this._loadDeploymentData();
        },

        _handleDeploymentFailed(data) {
            MessageBox.error(`Deployment failed: ${data.appName} - ${data.error}`);
            this._loadDeploymentData();
        }
    });
});