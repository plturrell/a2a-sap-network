sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel"
], function (ControllerExtension, MessageBox, MessageToast, Fragment, JSONModel) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent7.ext.controller.ObjectPageExt", {
        
        override: {
            onInit: function () {
                this._extensionAPI = this.base.getExtensionAPI();
            }
        },

        onStartAgent: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sAgentId = oContext.getProperty("managedAgent");
            var sAgentName = oContext.getProperty("taskName");
            
            MessageBox.confirm("Start agent '" + sAgentName + "'?", {
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        this._executeAgentOperation(sAgentId, "START");
                    }
                }.bind(this)
            });
        },

        onStopAgent: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sAgentId = oContext.getProperty("managedAgent");
            var sAgentName = oContext.getProperty("taskName");
            
            MessageBox.confirm("Stop agent '" + sAgentName + "'?", {
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        this._executeAgentOperation(sAgentId, "STOP");
                    }
                }.bind(this)
            });
        },

        onRestartAgent: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sAgentId = oContext.getProperty("managedAgent");
            var sAgentName = oContext.getProperty("taskName");
            
            MessageBox.confirm("Restart agent '" + sAgentName + "'?", {
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        this._executeAgentOperation(sAgentId, "RESTART");
                    }
                }.bind(this)
            });
        },

        _executeAgentOperation: function(sAgentId, sOperation) {
            this._extensionAPI.getView().setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent7/v1/agents/" + sAgentId + "/operations",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    operation: sOperation,
                    timestamp: new Date().toISOString()
                }),
                success: function(data) {
                    this._extensionAPI.getView().setBusy(false);
                    
                    var sMessage = sOperation.toLowerCase() + " operation initiated successfully";
                    if (data.estimatedTime) {
                        sMessage += "\\nEstimated completion: " + data.estimatedTime + " seconds";
                    }
                    
                    MessageToast.show(sMessage);
                    this._extensionAPI.refresh();
                    
                    if (data.monitoringUrl) {
                        this._startOperationMonitoring(data.operationId);
                    }
                }.bind(this),
                error: function(xhr) {
                    this._extensionAPI.getView().setBusy(false);
                    MessageBox.error("Operation failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        _startOperationMonitoring: function(sOperationId) {
            this._operationEventSource = new EventSource("/a2a/agent7/v1/operations/" + sOperationId + "/stream");
            
            this._operationEventSource.onmessage = function(event) {
                var data = JSON.parse(event.data);
                
                if (data.type === "progress") {
                    MessageToast.show(data.stage + ": " + data.progress + "%");
                } else if (data.type === "complete") {
                    this._operationEventSource.close();
                    this._extensionAPI.refresh();
                    MessageBox.success("Operation completed successfully!");
                } else if (data.type === "error") {
                    this._operationEventSource.close();
                    MessageBox.error("Operation failed: " + data.error);
                }
            }.bind(this);
            
            this._operationEventSource.onerror = function() {
                this._operationEventSource.close();
                MessageBox.error("Lost connection to operation monitoring");
            }.bind(this);
        },

        onUpdateAgent: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sAgentId = oContext.getProperty("managedAgent");
            
            if (!this._oUpdateDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent7.ext.fragment.UpdateAgent",
                    controller: this
                }).then(function(oDialog) {
                    this._oUpdateDialog = oDialog;
                    this.base.getView().addDependent(this._oUpdateDialog);
                    
                    var oModel = new JSONModel({
                        agentId: sAgentId,
                        updateType: "MINOR",
                        version: "",
                        rollbackEnabled: true,
                        restartAfterUpdate: true,
                        backupBeforeUpdate: true
                    });
                    this._oUpdateDialog.setModel(oModel, "update");
                    this._oUpdateDialog.open();
                    
                    this._loadAvailableVersions(sAgentId);
                }.bind(this));
            } else {
                this._oUpdateDialog.open();
                this._loadAvailableVersions(sAgentId);
            }
        },

        _loadAvailableVersions: function(sAgentId) {
            jQuery.ajax({
                url: "/a2a/agent7/v1/agents/" + sAgentId + "/available-versions",
                type: "GET",
                success: function(data) {
                    var oModel = this._oUpdateDialog.getModel("update");
                    var oData = oModel.getData();
                    oData.availableVersions = data.versions;
                    oData.currentVersion = data.currentVersion;
                    oModel.setData(oData);
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to load available versions: " + xhr.responseText);
                }.bind(this)
            });
        },

        onHealthCheck: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sAgentId = oContext.getProperty("managedAgent");
            
            this._extensionAPI.getView().setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent7/v1/agents/" + sAgentId + "/health-check",
                type: "POST",
                success: function(data) {
                    this._extensionAPI.getView().setBusy(false);
                    this._showHealthCheckResults(data);
                }.bind(this),
                error: function(xhr) {
                    this._extensionAPI.getView().setBusy(false);
                    MessageBox.error("Health check failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        _showHealthCheckResults: function(healthData) {
            var oView = this.base.getView();
            
            if (!this._oHealthResultsDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent7.ext.fragment.HealthCheckResults",
                    controller: this
                }).then(function(oDialog) {
                    this._oHealthResultsDialog = oDialog;
                    oView.addDependent(this._oHealthResultsDialog);
                    
                    var oModel = new JSONModel(healthData);
                    this._oHealthResultsDialog.setModel(oModel, "health");
                    this._oHealthResultsDialog.open();
                    
                    this._createHealthVisualizations(healthData);
                }.bind(this));
            } else {
                var oModel = new JSONModel(healthData);
                this._oHealthResultsDialog.setModel(oModel, "health");
                this._oHealthResultsDialog.open();
                this._createHealthVisualizations(healthData);
            }
        },

        _createHealthVisualizations: function(data) {
            var oRadarChart = this._oHealthResultsDialog.byId("healthRadarChart");
            if (!oRadarChart || !data.metrics) return;
            
            var aRadarData = Object.keys(data.metrics).map(function(key) {
                return {
                    Metric: key,
                    Score: data.metrics[key]
                };
            });
            
            var oChartModel = new sap.ui.model.json.JSONModel({
                healthData: aRadarData
            });
            oRadarChart.setModel(oChartModel);
        },

        onPerformanceCheck: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sAgentId = oContext.getProperty("managedAgent");
            
            this._extensionAPI.getView().setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent7/v1/agents/" + sAgentId + "/performance-check",
                type: "POST",
                success: function(data) {
                    this._extensionAPI.getView().setBusy(false);
                    this._showPerformanceResults(data);
                }.bind(this),
                error: function(xhr) {
                    this._extensionAPI.getView().setBusy(false);
                    MessageBox.error("Performance check failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        _showPerformanceResults: function(performanceData) {
            var sMessage = "Performance Check Results:\\n\\n";
            
            sMessage += "Response Time: " + performanceData.responseTime + "ms\\n";
            sMessage += "Throughput: " + performanceData.throughput + " req/sec\\n";
            sMessage += "Memory Usage: " + performanceData.memoryUsage + "%\\n";
            sMessage += "CPU Usage: " + performanceData.cpuUsage + "%\\n";
            sMessage += "Error Rate: " + performanceData.errorRate + "%\\n\\n";
            
            if (performanceData.recommendations && performanceData.recommendations.length > 0) {
                sMessage += "Recommendations:\\n";
                performanceData.recommendations.forEach(function(rec) {
                    sMessage += "• " + rec + "\\n";
                });
            }
            
            MessageBox.information(sMessage);
        },

        onConfigureAgent: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sAgentId = oContext.getProperty("managedAgent");
            
            if (!this._oConfigDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent7.ext.fragment.AgentConfiguration",
                    controller: this
                }).then(function(oDialog) {
                    this._oConfigDialog = oDialog;
                    this.base.getView().addDependent(this._oConfigDialog);
                    this._oConfigDialog.open();
                    this._loadAgentConfiguration(sAgentId);
                }.bind(this));
            } else {
                this._oConfigDialog.open();
                this._loadAgentConfiguration(sAgentId);
            }
        },

        _loadAgentConfiguration: function(sAgentId) {
            jQuery.ajax({
                url: "/a2a/agent7/v1/agents/" + sAgentId + "/configuration",
                type: "GET",
                success: function(data) {
                    var oModel = new JSONModel({
                        agentId: sAgentId,
                        configuration: data.configuration,
                        schema: data.schema,
                        modifiedFields: []
                    });
                    this._oConfigDialog.setModel(oModel, "config");
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to load configuration: " + xhr.responseText);
                }.bind(this)
            });
        },

        onSaveConfiguration: function() {
            var oModel = this._oConfigDialog.getModel("config");
            var oData = oModel.getData();
            
            this._oConfigDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent7/v1/agents/" + oData.agentId + "/configuration",
                type: "PUT",
                contentType: "application/json",
                data: JSON.stringify({
                    configuration: oData.configuration,
                    modifiedFields: oData.modifiedFields
                }),
                success: function(data) {
                    this._oConfigDialog.setBusy(false);
                    this._oConfigDialog.close();
                    
                    MessageBox.success(
                        "Configuration updated successfully!\\n" +
                        "Restart required: " + (data.restartRequired ? "Yes" : "No")
                    );
                    
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    this._oConfigDialog.setBusy(false);
                    MessageBox.error("Configuration update failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        onViewLogs: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sAgentId = oContext.getProperty("managedAgent");
            
            if (!this._oLogsDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent7.ext.fragment.AgentLogs",
                    controller: this
                }).then(function(oDialog) {
                    this._oLogsDialog = oDialog;
                    this.base.getView().addDependent(this._oLogsDialog);
                    this._oLogsDialog.open();
                    this._loadAgentLogs(sAgentId);
                }.bind(this));
            } else {
                this._oLogsDialog.open();
                this._loadAgentLogs(sAgentId);
            }
        },

        _loadAgentLogs: function(sAgentId) {
            jQuery.ajax({
                url: "/a2a/agent7/v1/agents/" + sAgentId + "/logs",
                type: "GET",
                data: {
                    limit: 1000,
                    level: "ALL"
                },
                success: function(data) {
                    var oModel = new JSONModel({
                        logs: data.logs,
                        logLevels: data.availableLevels,
                        selectedLevel: "ALL"
                    });
                    this._oLogsDialog.setModel(oModel, "logs");
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to load logs: " + xhr.responseText);
                }.bind(this)
            });
        },

        onAgentDetails: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sAgentId = oContext.getProperty("managedAgent");
            
            jQuery.ajax({
                url: "/a2a/agent7/v1/agents/" + sAgentId + "/details",
                type: "GET",
                success: function(data) {
                    this._showAgentDetails(data);
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to load agent details: " + xhr.responseText);
                }.bind(this)
            });
        },

        _showAgentDetails: function(agentData) {
            var sMessage = "Agent Details:\\n\\n";
            
            sMessage += "Name: " + agentData.name + "\\n";
            sMessage += "Type: " + agentData.type + "\\n";
            sMessage += "Version: " + agentData.version + "\\n";
            sMessage += "Status: " + agentData.status + "\\n";
            sMessage += "Endpoint: " + agentData.endpoint + "\\n";
            sMessage += "Uptime: " + agentData.uptime + "\\n";
            sMessage += "Last Health Check: " + agentData.lastHealthCheck + "\\n\\n";
            
            if (agentData.capabilities && agentData.capabilities.length > 0) {
                sMessage += "Capabilities:\\n";
                agentData.capabilities.forEach(function(cap) {
                    sMessage += "• " + cap + "\\n";
                });
                sMessage += "\\n";
            }
            
            if (agentData.dependencies && agentData.dependencies.length > 0) {
                sMessage += "Dependencies:\\n";
                agentData.dependencies.forEach(function(dep) {
                    sMessage += "• " + dep + "\\n";
                });
            }
            
            MessageBox.information(sMessage);
        },

        onCoordination: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sAgentId = oContext.getProperty("managedAgent");
            
            jQuery.ajax({
                url: "/a2a/agent7/v1/agents/" + sAgentId + "/coordination-status",
                type: "GET",
                success: function(data) {
                    this._showCoordinationStatus(data);
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to load coordination status: " + xhr.responseText);
                }.bind(this)
            });
        },

        _showCoordinationStatus: function(coordinationData) {
            var sMessage = "Agent Coordination Status:\\n\\n";
            
            sMessage += "Coordination Enabled: " + (coordinationData.enabled ? "Yes" : "No") + "\\n";
            sMessage += "Active Connections: " + coordinationData.activeConnections + "\\n";
            sMessage += "Workflow Participation: " + coordinationData.workflowParticipation + "\\n";
            sMessage += "Load Balancing: " + (coordinationData.loadBalancing ? "Enabled" : "Disabled") + "\\n";
            sMessage += "Trust Level: " + coordinationData.trustLevel + "\\n\\n";
            
            if (coordinationData.connectedAgents && coordinationData.connectedAgents.length > 0) {
                sMessage += "Connected Agents:\\n";
                coordinationData.connectedAgents.forEach(function(agent) {
                    sMessage += "• " + agent.name + " (" + agent.status + ")\\n";
                });
            }
            
            MessageBox.information(sMessage, {
                actions: ["Enable Coordination", "Disable Coordination", MessageBox.Action.CLOSE],
                onClose: function(oAction) {
                    if (oAction === "Enable Coordination" || oAction === "Disable Coordination") {
                        this._toggleCoordination(sAgentId, oAction === "Enable Coordination");
                    }
                }.bind(this)
            });
        },

        _toggleCoordination: function(sAgentId, bEnable) {
            jQuery.ajax({
                url: "/a2a/agent7/v1/agents/" + sAgentId + "/coordination",
                type: "PUT",
                contentType: "application/json",
                data: JSON.stringify({ enabled: bEnable }),
                success: function(data) {
                    MessageToast.show("Coordination " + (bEnable ? "enabled" : "disabled") + " successfully");
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to toggle coordination: " + xhr.responseText);
                }.bind(this)
            });
        },

        onConfirmUpdate: function() {
            var oModel = this._oUpdateDialog.getModel("update");
            var oData = oModel.getData();
            
            if (!oData.version) {
                MessageBox.error("Please select a version to update to");
                return;
            }
            
            this._oUpdateDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent7/v1/agents/" + oData.agentId + "/update",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify(oData),
                success: function(data) {
                    this._oUpdateDialog.setBusy(false);
                    this._oUpdateDialog.close();
                    
                    MessageBox.success(
                        "Agent update initiated!\\n" +
                        "Update ID: " + data.updateId + "\\n" +
                        "Estimated time: " + data.estimatedTime + " minutes"
                    );
                    
                    this._extensionAPI.refresh();
                    this._startOperationMonitoring(data.updateId);
                }.bind(this),
                error: function(xhr) {
                    this._oUpdateDialog.setBusy(false);
                    MessageBox.error("Agent update failed: " + xhr.responseText);
                }.bind(this)
            });
        }
    });
});