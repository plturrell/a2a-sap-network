sap.ui.define([
    "./BaseController",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/model/json/JSONModel",
    "sap/base/Log"
], function(BaseController, MessageToast, MessageBox, JSONModel, Log) {
    "use strict";

    return BaseController.extend("a2a.network.fiori.controller.Operations", {

        getCurrentUserEmail() {
            try {
                return sap.ushell.Container.getUser().getEmail() ||
                       this.getOwnerComponent().getModel("app").getProperty("/currentUser/email") ||
                       "user@a2a.network";
            } catch (error) {
                return "user@a2a.network";
            }
        },

        onInit() {
            // Initialize operations model
            const oOperationsModel = new JSONModel({
                systemHealth: {
                    status: "healthy",
                    uptime: "0h 0m",
                    version: "1.0.0",
                    lastRestart: new Date()
                },
                performanceMetrics: {
                    cpuUsage: 25,
                    memoryUsage: 45,
                    diskUsage: 30,
                    networkLatency: 12
                },
                alerts: [],
                recentOperations: [],
                services: [
                    {
                        name: "A2A Service",
                        status: "running",
                        port: 4004,
                        uptime: "2h 30m",
                        requests: 1234,
                        errors: 2
                    },
                    {
                        name: "Blockchain Service",
                        status: "running",
                        port: 8545,
                        uptime: "2h 30m",
                        requests: 456,
                        errors: 0
                    },
                    {
                        name: "Message Router",
                        status: "running",
                        port: 3000,
                        uptime: "2h 30m",
                        requests: 789,
                        errors: 1
                    }
                ]
            });
            this.getView().setModel(oOperationsModel, "operations");

            // Load operations data
            this._loadOperationsData();

            // Set up auto-refresh
            this._setupAutoRefresh();

            Log.info("Operations controller initialized");
        },

        async _loadOperationsData() {
            this.showBusyIndicator();

            try {
                // Load system health
                const healthResponse = await fetch("/health");
                if (healthResponse.ok) {
                    const healthData = await healthResponse.json();
                    this._updateSystemHealth(healthData);
                }

                // Load performance metrics
                const metricsResponse = await fetch("/metrics");
                if (metricsResponse.ok) {
                    const metricsText = await metricsResponse.text();
                    this._updatePerformanceMetrics(metricsText);
                }

                // Load operations service data
                const opsResponse = await fetch("/api/v1/operations/status");
                if (opsResponse.ok) {
                    const opsData = await opsResponse.json();
                    this._updateOperationsData(opsData);
                }

            } catch (error) {
                Log.error("Failed to load operations data", error);
                MessageToast.show(this.getResourceBundle().getText("operationsLoadError"));
            } finally {
                this.hideBusyIndicator();
            }
        },

        _updateSystemHealth(healthData) {
            const oModel = this.getView().getModel("operations");
            oModel.setProperty("/systemHealth", {
                status: healthData.status || "unknown",
                uptime: healthData.uptime || "0h 0m",
                version: healthData.version || "1.0.0",
                lastRestart: healthData.lastRestart ? new Date(healthData.lastRestart) : new Date(),
                nodeVersion: healthData.nodeVersion || process.version,
                environment: healthData.environment || "development"
            });
        },

        _updatePerformanceMetrics(metricsText) {
            // Parse Prometheus format metrics
            const metrics = this._parsePrometheusMetrics(metricsText);

            const oModel = this.getView().getModel("operations");
            oModel.setProperty("/performanceMetrics", {
                cpuUsage: metrics.cpu_usage || 0,
                memoryUsage: metrics.memory_usage_percent || 0,
                diskUsage: metrics.disk_usage_percent || 0,
                networkLatency: metrics.network_latency_ms || 0,
                requestsPerSecond: metrics.requests_per_second || 0,
                errorsPerMinute: metrics.errors_per_minute || 0
            });
        },

        _parsePrometheusMetrics(metricsText) {
            const metrics = {};
            const lines = metricsText.split("\n");

            lines.forEach(line => {
                if (line.startsWith("#") || !line.includes(" ")) {
                    return;
                }

                const [nameWithLabels, value] = line.split(" ");
                const metricName = nameWithLabels.split("{")[0];
                metrics[metricName] = parseFloat(value) || 0;
            });

            return metrics;
        },

        _updateOperationsData(opsData) {
            const oModel = this.getView().getModel("operations");

            if (opsData.alerts) {
                oModel.setProperty("/alerts", opsData.alerts);
            }

            if (opsData.recentOperations) {
                oModel.setProperty("/recentOperations", opsData.recentOperations);
            }

            if (opsData.services) {
                oModel.setProperty("/services", opsData.services);
            }
        },

        _setupAutoRefresh() {
            // Refresh every 30 seconds
            this._refreshInterval = setInterval(() => {
                this._loadOperationsData();
            }, 30000);
        },

        onRefreshHealth() {
            this._loadOperationsData();
        },

        onSystemRestart() {
            MessageBox.confirm(
                this.getResourceBundle().getText("confirmSystemRestart"),
                {
                    title: this.getResourceBundle().getText("systemRestart"),
                    onClose: (sAction) => {
                        if (sAction === MessageBox.Action.OK) {
                            this._restartSystem();
                        }
                    }
                }
            );
        },

        async _restartSystem() {
            try {
                const response = await fetch("/api/v1/operations/restart", {
                    method: "POST"
                });

                if (response.ok) {
                    MessageToast.show(this.getResourceBundle().getText("systemRestartInitiated"));
                } else {
                    MessageBox.error(this.getResourceBundle().getText("systemRestartFailed"));
                }

            } catch (error) {
                Log.error("Failed to restart system", error);
                MessageBox.error(this.getResourceBundle().getText("systemRestartError"));
            }
        },

        onClearLogs() {
            MessageBox.confirm(
                this.getResourceBundle().getText("confirmClearLogs"),
                {
                    title: this.getResourceBundle().getText("clearLogs"),
                    onClose: (sAction) => {
                        if (sAction === MessageBox.Action.OK) {
                            this._clearLogs();
                        }
                    }
                }
            );
        },

        async _clearLogs() {
            try {
                const response = await fetch("/api/v1/operations/logs", {
                    method: "DELETE"
                });

                if (response.ok) {
                    MessageToast.show(this.getResourceBundle().getText("logsCleared"));
                } else {
                    MessageBox.error(this.getResourceBundle().getText("logsClearFailed"));
                }

            } catch (error) {
                Log.error("Failed to clear logs", error);
                MessageBox.error(this.getResourceBundle().getText("logsClearError"));
            }
        },

        async onDownloadLogs() {
            try {
                const response = await fetch("/api/v1/operations/logs/download");

                if (response.ok) {
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement("a");
                    a.href = url;
                    a.download = `a2a-logs-${new Date().toISOString().split("T")[0]}.zip`;
                    document.body.appendChild(a);
                    a.click();
                    window.URL.revokeObjectURL(url);
                    document.body.removeChild(a);

                    MessageToast.show(this.getResourceBundle().getText("logsDownloaded"));
                } else {
                    MessageBox.error(this.getResourceBundle().getText("logsDownloadFailed"));
                }

            } catch (error) {
                Log.error("Failed to download logs", error);
                MessageBox.error(this.getResourceBundle().getText("logsDownloadError"));
            }
        },

        onServiceAction(oEvent) {
            const oSource = oEvent.getSource();
            const sAction = oSource.data("action");
            const _oContext = oSource.getBindingContext("operations");
            const oService = oContext.getObject();

            switch (sAction) {
            case "restart":
                this._restartService(oService.name);
                break;
            case "stop":
                this._stopService(oService.name);
                break;
            case "logs":
                this._viewServiceLogs(oService.name);
                break;
            case "configure":
                this._configureService(oService.name);
                break;
            case "scale":
                this._scaleService(oService.name);
                break;
            default:
                MessageToast.show(this.getResourceBundle().getText("actionNotSupported", [sAction]));
            }
        },

        async _restartService(sServiceName) {
            try {
                const response = await fetch(`/api/v1/operations/services/${sServiceName}/restart`, {
                    method: "POST"
                });

                if (response.ok) {
                    MessageToast.show(this.getResourceBundle().getText("serviceRestarted", [sServiceName]));
                    this._loadOperationsData();
                } else {
                    MessageBox.error(this.getResourceBundle().getText("serviceRestartFailed"));
                }

            } catch (error) {
                Log.error("Failed to restart service", error);
                MessageBox.error(this.getResourceBundle().getText("serviceRestartError"));
            }
        },

        async _stopService(sServiceName) {
            MessageBox.confirm(
                this.getResourceBundle().getText("confirmStopService", [sServiceName]),
                {
                    title: this.getResourceBundle().getText("stopService"),
                    onClose: async(sAction) => {
                        if (sAction === MessageBox.Action.OK) {
                            try {
                                const response = await fetch(`/api/v1/operations/services/${sServiceName}/stop`, {
                                    method: "POST"
                                });

                                if (response.ok) {
                                    MessageToast.show(this.getResourceBundle().getText("serviceStopped", [sServiceName]));
                                    this._loadOperationsData();
                                } else {
                                    MessageBox.error(this.getResourceBundle().getText("serviceStopFailed"));
                                }

                            } catch (error) {
                                Log.error("Failed to stop service", error);
                                MessageBox.error(this.getResourceBundle().getText("serviceStopError"));
                            }
                        }
                    }
                }
            );
        },

        _viewServiceLogs(sServiceName) {
            // This could open a dialog with service logs or navigate to a logs view
            MessageBox.information(
                this.getResourceBundle().getText("serviceLogsNotImplemented"),
                {
                    title: this.getResourceBundle().getText("serviceLogs", [sServiceName])
                }
            );
        },

        onAlertAction(oEvent) {
            const oSource = oEvent.getSource();
            const sAction = oSource.data("action");
            const _oContext = oSource.getBindingContext("operations");
            const oAlert = oContext.getObject();

            switch (sAction) {
            case "acknowledge":
                this._acknowledgeAlert(oAlert.id);
                break;
            case "dismiss":
                this._dismissAlert(oAlert.id);
                break;
            case "resolve":
                this._resolveAlert(oAlert.id);
                break;
            case "escalate":
                this._escalateAlert(oAlert.id);
                break;
            default:
                MessageToast.show(this.getResourceBundle().getText("alertActionNotSupported", [sAction]));
            }
        },

        async _acknowledgeAlert(sAlertId) {
            try {
                const response = await fetch(`/api/v1/operations/alerts/${sAlertId}/acknowledge`, {
                    method: "POST"
                });

                if (response.ok) {
                    MessageToast.show(this.getResourceBundle().getText("alertAcknowledged"));
                    this._loadOperationsData();
                } else {
                    MessageBox.error(this.getResourceBundle().getText("alertAcknowledgeFailed"));
                }

            } catch (error) {
                Log.error("Failed to acknowledge alert", error);
                MessageBox.error(this.getResourceBundle().getText("alertAcknowledgeError"));
            }
        },

        async _dismissAlert(sAlertId) {
            try {
                const response = await fetch(`/api/v1/operations/alerts/${sAlertId}`, {
                    method: "DELETE"
                });

                if (response.ok) {
                    MessageToast.show(this.getResourceBundle().getText("alertDismissed"));
                    this._loadOperationsData();
                } else {
                    MessageBox.error(this.getResourceBundle().getText("alertDismissFailed"));
                }

            } catch (error) {
                Log.error("Failed to dismiss alert", error);
                MessageBox.error(this.getResourceBundle().getText("alertDismissError"));
            }
        },

        onOperationDetails(oEvent) {
            const _oContext = oEvent.getSource().getBindingContext("operations");
            const oOperation = oContext.getObject();

            // Show operation details in a dialog or navigate to details view
            const sDetails = `
Operation: ${oOperation.type}
Status: ${oOperation.status}
Started: ${new Date(oOperation.startTime).toLocaleString()}
Duration: ${oOperation.duration}
User: ${oOperation.user}
Details: ${oOperation.details}
            `;

            MessageBox.information(sDetails, {
                title: this.getResourceBundle().getText("operationDetails")
            });
        },

        onExport() {
            // Export operations data
            const oModel = this.getView().getModel("operations");
            const oData = oModel.getData();

            const exportData = {
                timestamp: new Date().toISOString(),
                systemHealth: oData.systemHealth,
                performanceMetrics: oData.performanceMetrics,
                services: oData.services,
                alerts: oData.alerts,
                recentOperations: oData.recentOperations
            };

            // Create download
            const element = document.createElement("a");
            element.setAttribute("href", `data:text/json;charset=utf-8,${ encodeURIComponent(JSON.stringify(exportData, null, 2))}`);
            element.setAttribute("download", `a2a-operations-${new Date().toISOString().split("T")[0]}.json`);
            element.style.display = "none";
            document.body.appendChild(element);
            element.click();
            document.body.removeChild(element);

            MessageToast.show(this.getResourceBundle().getText("operationsDataExported"));
        },

        _configureService(sServiceName) {
            // Open service configuration dialog
            MessageToast.show(this.getResourceBundle().getText("openingServiceConfiguration", [sServiceName]));
            // In production, open configuration dialog
        },

        _scaleService(sServiceName) {
            // Open service scaling dialog
            if (!this._oScaleDialog) {
                MessageBox.confirm(
                    this.getResourceBundle().getText("scaleServiceConfirm", [sServiceName]),
                    {
                        title: this.getResourceBundle().getText("scaleService"),
                        actions: [MessageBox.Action.OK, MessageBox.Action.CANCEL],
                        onClose: function(sAction) {
                            if (sAction === MessageBox.Action.OK) {
                                // In production, call scaling API
                                MessageToast.show(this.getResourceBundle().getText("serviceScalingInitiated", [sServiceName]));
                            }
                        }.bind(this)
                    }
                );
            }
        },

        async _resolveAlert(sAlertId) {
            try {
                const response = await fetch(`/api/v1/operations/alerts/${sAlertId}/resolve`, {
                    method: "POST",
                    headers: {
                        "Content-Type": "application/json"
                    },
                    body: JSON.stringify({
                        resolvedBy: this.getCurrentUserEmail(),
                        resolution: "Issue resolved"
                    })
                });

                if (response.ok) {
                    MessageToast.show(this.getResourceBundle().getText("alertResolved"));
                    this._loadOperationsData();
                } else {
                    MessageToast.show(this.getResourceBundle().getText("alertResolveError"));
                }
            } catch (error) {
                MessageToast.show(this.getResourceBundle().getText("alertResolveError"));
            }
        },

        async _escalateAlert(sAlertId) {
            MessageBox.confirm(
                this.getResourceBundle().getText("escalateAlertConfirm"),
                {
                    title: this.getResourceBundle().getText("escalateAlert"),
                    onClose: async function(sAction) {
                        if (sAction === MessageBox.Action.OK) {
                            try {
                                const response = await fetch(`/api/v1/operations/alerts/${sAlertId}/escalate`, {
                                    method: "POST",
                                    headers: {
                                        "Content-Type": "application/json"
                                    },
                                    body: JSON.stringify({
                                        escalatedBy: this.getCurrentUserEmail(),
                                        escalationLevel: 2
                                    })
                                });

                                if (response.ok) {
                                    MessageToast.show(this.getResourceBundle().getText("alertEscalated"));
                                    this._loadOperationsData();
                                } else {
                                    MessageToast.show(this.getResourceBundle().getText("alertEscalateError"));
                                }
                            } catch (error) {
                                MessageToast.show(this.getResourceBundle().getText("alertEscalateError"));
                            }
                        }
                    }.bind(this)
                }
            );
        },

        onExit() {
            // Clean up auto-refresh interval
            if (this._refreshInterval) {
                clearInterval(this._refreshInterval);
            }
        }
    });
});