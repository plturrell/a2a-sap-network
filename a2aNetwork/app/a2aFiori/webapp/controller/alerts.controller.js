sap.ui.define([
    "./BaseController",
    "sap/ui/model/json/JSONModel",
    "sap/ui/model/Filter",
    "sap/ui/model/FilterOperator",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/export/Spreadsheet",
    "sap/ui/export/library",
    "../utils/websocket"
], function (BaseController, JSONModel, Filter, FilterOperator, MessageToast, MessageBox, Spreadsheet, exportLibrary, WebSocketUtil) {
    "use strict";

    var EdmType = exportLibrary.EdmType;

    return BaseController.extend("a2a.network.fiori.controller.Alerts", {

        onInit: function () {
            BaseController.prototype.onInit.apply(this, arguments);
            
            // Initialize models
            this._initializeModels();
            
            // Set up real-time alert streaming
            this._setupRealtimeAlerts();
            
            // Load initial data
            this._loadAlerts();
            
            // Start monitoring for new alerts
            this._startAlertMonitoring();
            
            // Load notification settings
            this._loadNotificationSettings();
        },

        _initializeModels: function() {
            // Alerts model
            this.oAlertsModel = new JSONModel({
                activeAlerts: [],
                alertHistory: [],
                alertRules: [],
                selectedAlert: null,
                selectedRule: null,
                statistics: {
                    critical: 0,
                    high: 0,
                    medium: 0,
                    low: 0,
                    total: 0
                }
            });
            this.getView().setModel(this.oAlertsModel, "alerts");
            
            // Settings model
            this.oSettingsModel = new JSONModel({
                notifications: {
                    email: { enabled: true, address: "admin@a2a-network.com" },
                    sms: { enabled: false, number: "" },
                    push: { enabled: true },
                    slack: { enabled: true, webhook: "https://hooks.slack.com/services/..." },
                    severityThreshold: 1 // 0=all, 1=medium+, 2=high+, 3=critical
                }
            });
            this.getView().setModel(this.oSettingsModel, "settings");
            
            // Update UI model
            this.oUIModel.setProperty("/alertView", "active");
            this.oUIModel.setProperty("/showActiveRulesOnly", true);
            this.oUIModel.setProperty("/showRealtimeAlert", false);
        },

        _setupRealtimeAlerts: function() {
            // Modern WebSocket connection for real-time alerts using socket.io
            this._connectWebSocket().then(() => {
                // Subscribe to alert-related topics
                WebSocketUtil.subscribe([
                    'monitoring.alerts',
                    'system.health',
                    'agent.events',
                    'service.events'
                ]).then((data) => {
                    console.log("Subscribed to alert topics:", data);
                }).catch((error) => {
                    console.error("Failed to subscribe to alert topics:", error);
                });

                // Listen for real-time alert events
                WebSocketUtil.on('alert.new', this._handleRealtimeAlert.bind(this));
                WebSocketUtil.on('alert.updated', this._handleAlertUpdate.bind(this));
                WebSocketUtil.on('system.health.changed', this._handleSystemHealthChange.bind(this));
                
            }).catch((error) => {
                console.error("Failed to establish WebSocket connection:", error);
                // Fallback: retry connection after 5 seconds
                setTimeout(this._setupRealtimeAlerts.bind(this), 5000);
            });
        },

        _connectWebSocket: async function() {
            // Get authentication token from user model
            const userModel = this.getOwnerComponent().getModel("user");
            const token = userModel?.getProperty("/token") || "dev-token";

            return WebSocketUtil.connect({ token: token });
        },

        _handleAlertUpdate: function(alertData) {
            // Update existing alert in the model
            var aActiveAlerts = this.oAlertsModel.getProperty("/activeAlerts");
            var alertIndex = aActiveAlerts.findIndex(alert => alert.id === alertData.id);
            
            if (alertIndex !== -1) {
                aActiveAlerts[alertIndex] = { ...aActiveAlerts[alertIndex], ...alertData };
                this.oAlertsModel.setProperty("/activeAlerts", aActiveAlerts);
                this._updateStatistics();
            }
        },

        _handleSystemHealthChange: function(healthData) {
            // Update system health indicators and create alerts if needed
            if (healthData.status === 'degraded' || healthData.status === 'critical') {
                var alert = {
                    id: "health_alert_" + Date.now(),
                    severity: healthData.status === 'critical' ? 'critical' : 'high',
                    priority: healthData.status === 'critical' ? 1 : 2,
                    title: `System Health ${healthData.status.charAt(0).toUpperCase() + healthData.status.slice(1)}`,
                    description: `System health has changed to ${healthData.status}. ${healthData.message || ''}`,
                    source: "SystemHealth",
                    timestamp: new Date(healthData.timestamp || Date.now()),
                    acknowledged: false,
                    metrics: healthData.metrics || {}
                };
                
                this._handleRealtimeAlert(alert);
            }
        },

        _handleRealtimeAlert: function(alert) {
            // Add to active alerts
            var aActiveAlerts = this.oAlertsModel.getProperty("/activeAlerts");
            aActiveAlerts.unshift(alert);
            this.oAlertsModel.setProperty("/activeAlerts", aActiveAlerts);
            
            // Update statistics
            this._updateStatistics();
            
            // Show notification
            this._showRealtimeNotification(alert);
            
            // Play sound for critical alerts
            if (alert.severity === "critical") {
                this._playAlertSound();
            }
            
            // Send notifications based on settings
            this._sendNotifications(alert);
        },

        _showRealtimeNotification: function(alert) {
            var sMessage = alert.title + " - " + alert.description;
            var sType = alert.severity === "critical" ? "Error" : 
                       alert.severity === "high" ? "Warning" : "Information";
            
            this.oUIModel.setProperty("/showRealtimeAlert", true);
            this.oUIModel.setProperty("/realtimeAlertMessage", sMessage);
            this.oUIModel.setProperty("/realtimeAlertType", sType);
            
            // Auto-hide after 10 seconds
            setTimeout(function() {
                this.oUIModel.setProperty("/showRealtimeAlert", false);
            }.bind(this), 10000);
        },

        _playAlertSound: function() {
            // Play alert sound for critical alerts
            try {
                var audio = new Audio("/sounds/alert.mp3");
                audio.play();
            } catch (e) {
                console.error("Failed to play alert sound:", e);
            }
        },

        _loadAlerts: function() {
            this.showSkeletonLoading(this.getResourceBundle().getText("alerts.loading"));
            
            // Simulate loading alerts - in production, call backend service
            setTimeout(function() {
                var aActiveAlerts = this._generateActiveAlerts();
                var aAlertHistory = this._generateAlertHistory();
                var aAlertRules = this._generateAlertRules();
                
                this.oAlertsModel.setProperty("/activeAlerts", aActiveAlerts);
                this.oAlertsModel.setProperty("/alertHistory", aAlertHistory);
                this.oAlertsModel.setProperty("/alertRules", aAlertRules);
                
                this._updateStatistics();
                this.hideLoading();
            }.bind(this), 1000);
        },

        _generateActiveAlerts: function() {
            var aAlerts = [];
            var aSeverities = ["critical", "high", "medium", "low"];
            var aSources = ["AgentMonitor", "BlockchainService", "SystemHealth", "SecurityScanner", "PerformanceMonitor"];
            
            // Generate 5-10 active alerts
            var alertCount = 5 + Math.floor(Math.random() * 6);
            for (var i = 0; i < alertCount; i++) {
                aAlerts.push({
                    id: "alert_" + Date.now() + "_" + i,
                    severity: aSeverities[Math.floor(Math.random() * aSeverities.length)],
                    priority: Math.floor(Math.random() * 4),
                    title: this._generateAlertTitle(),
                    description: this._generateAlertDescription(),
                    source: aSources[Math.floor(Math.random() * aSources.length)],
                    timestamp: new Date(Date.now() - Math.random() * 3600000),
                    acknowledged: Math.random() > 0.7,
                    metrics: {
                        cpu: Math.floor(Math.random() * 100),
                        memory: Math.floor(Math.random() * 100),
                        responseTime: Math.floor(Math.random() * 2000)
                    }
                });
            }
            
            return aAlerts;
        },

        _generateAlertTitle: function() {
            var titles = [
                "High CPU Usage Detected",
                "Memory Threshold Exceeded",
                "Blockchain Connection Lost",
                "Agent Response Time Degradation",
                "Security Vulnerability Detected",
                "Database Connection Pool Exhausted",
                "API Rate Limit Approaching",
                "Smart Contract Execution Failed",
                "Network Latency Spike",
                "Disk Space Running Low"
            ];
            return titles[Math.floor(Math.random() * titles.length)];
        },

        _generateAlertDescription: function() {
            var descriptions = [
                "CPU usage has exceeded 90% for more than 5 minutes on node agent-01",
                "Memory consumption reached 95% on the blockchain service instance",
                "Unable to establish connection to blockchain node at endpoint https://eth-node.a2a.network",
                "Average agent response time increased to 2.5 seconds (threshold: 1 second)",
                "Critical security patch required for dependency package web3@4.2.0",
                "Database connection pool limit reached (100/100 connections in use)",
                "API endpoint /agents/execute approaching rate limit (950/1000 requests)",
                "Smart contract deployment failed due to insufficient gas: required 300000, available 250000",
                "Network latency to region us-east-1 increased to 250ms (normal: 50ms)",
                "Disk usage on /data partition reached 85% (17GB free of 100GB)"
            ];
            return descriptions[Math.floor(Math.random() * descriptions.length)];
        },

        _generateAlertHistory: function() {
            var aHistory = [];
            var aStatuses = ["resolved", "acknowledged", "escalated"];
            var aResolvers = ["admin@sap.com", "operations@sap.com", "system", "auto-resolved"];
            
            // Generate 100 historical alerts
            for (var i = 0; i < 100; i++) {
                var timestamp = new Date(Date.now() - Math.random() * 30 * 24 * 3600000);
                var resolvedTime = new Date(timestamp.getTime() + Math.random() * 3600000);
                
                aHistory.push({
                    id: "hist_alert_" + i,
                    severity: ["critical", "high", "medium", "low"][Math.floor(Math.random() * 4)],
                    title: this._generateAlertTitle(),
                    source: ["AgentMonitor", "BlockchainService", "SystemHealth"][Math.floor(Math.random() * 3)],
                    timestamp: timestamp,
                    status: aStatuses[Math.floor(Math.random() * aStatuses.length)],
                    resolvedBy: aResolvers[Math.floor(Math.random() * aResolvers.length)],
                    duration: Math.floor((resolvedTime - timestamp) / 60000) + " min",
                    actionsTaken: [
                        { action: "Alert triggered", user: "system", timestamp: timestamp },
                        { action: "Acknowledged", user: "admin@sap.com", timestamp: new Date(timestamp.getTime() + 300000) },
                        { action: "Resolved", user: aResolvers[Math.floor(Math.random() * aResolvers.length)], timestamp: resolvedTime }
                    ]
                });
            }
            
            return aHistory;
        },

        _generateAlertRules: function() {
            return [
                {
                    id: "rule_001",
                    name: "High CPU Usage Alert",
                    description: "Triggers when CPU usage exceeds threshold",
                    type: "Threshold",
                    severity: "high",
                    active: true,
                    condition: "cpu > 80%",
                    conditionDisplay: "CPU Usage > 80% for 5 minutes",
                    lastTriggered: new Date(Date.now() - 3600000),
                    triggerCount: 15,
                    avgResponseTime: "8 min",
                    notifications: [
                        { type: "email", recipient: "ops-team@sap.com" },
                        { type: "slack", recipient: "#alerts-channel" }
                    ]
                },
                {
                    id: "rule_002",
                    name: "Blockchain Connection Monitor",
                    description: "Alerts on blockchain connectivity issues",
                    type: "Availability",
                    severity: "critical",
                    active: true,
                    condition: "blockchain.connected === false",
                    conditionDisplay: "Blockchain connection lost",
                    lastTriggered: new Date(Date.now() - 86400000),
                    triggerCount: 3,
                    avgResponseTime: "2 min",
                    notifications: [
                        { type: "email", recipient: "blockchain-team@sap.com" },
                        { type: "sms", recipient: "+1234567890" },
                        { type: "push", recipient: "all" }
                    ]
                },
                {
                    id: "rule_003",
                    name: "Memory Leak Detection",
                    description: "Detects potential memory leaks",
                    type: "Anomaly",
                    severity: "medium",
                    active: true,
                    condition: "memory.growth > 10% per hour",
                    conditionDisplay: "Memory growth > 10% per hour",
                    lastTriggered: new Date(Date.now() - 172800000),
                    triggerCount: 8,
                    avgResponseTime: "15 min",
                    notifications: [
                        { type: "email", recipient: "dev-team@sap.com" }
                    ]
                },
                {
                    id: "rule_004",
                    name: "Agent Response Time",
                    description: "Monitors agent execution performance",
                    type: "Performance",
                    severity: "medium",
                    active: false,
                    condition: "agent.responseTime > 1000ms",
                    conditionDisplay: "Agent response time > 1 second",
                    lastTriggered: null,
                    triggerCount: 0,
                    avgResponseTime: "N/A",
                    notifications: [
                        { type: "email", recipient: "agent-team@sap.com" }
                    ]
                },
                {
                    id: "rule_005",
                    name: "Security Scan Alert",
                    description: "Alerts on security vulnerabilities",
                    type: "Security",
                    severity: "critical",
                    active: true,
                    condition: "security.vulnerabilities > 0",
                    conditionDisplay: "Security vulnerabilities detected",
                    lastTriggered: new Date(Date.now() - 604800000),
                    triggerCount: 2,
                    avgResponseTime: "30 min",
                    notifications: [
                        { type: "email", recipient: "security-team@sap.com" },
                        { type: "slack", recipient: "#security-alerts" },
                        { type: "push", recipient: "security-admins" }
                    ]
                }
            ];
        },

        _updateStatistics: function() {
            var aActiveAlerts = this.oAlertsModel.getProperty("/activeAlerts");
            var oStats = {
                critical: 0,
                high: 0,
                medium: 0,
                low: 0,
                total: aActiveAlerts.length
            };
            
            aActiveAlerts.forEach(function(alert) {
                oStats[alert.severity]++;
            });
            
            this.oAlertsModel.setProperty("/statistics", oStats);
        },

        onSearchAlerts: function(oEvent) {
            var sQuery = oEvent.getParameter("query");
            var oList = this.byId("activeAlertsList");
            var oBinding = oList.getBinding("items");
            
            if (sQuery) {
                var aFilters = [
                    new Filter("title", FilterOperator.Contains, sQuery),
                    new Filter("description", FilterOperator.Contains, sQuery),
                    new Filter("source", FilterOperator.Contains, sQuery)
                ];
                oBinding.filter(new Filter({
                    filters: aFilters,
                    and: false
                }));
            } else {
                oBinding.filter([]);
            }
        },

        onFilterAlerts: function() {
            // Open filter dialog
            if (!this._oFilterDialog) {
                this._oFilterDialog = sap.ui.xmlfragment(
                    "a2a.network.fiori.fragment.AlertFilterDialog",
                    this
                );
                this.getView().addDependent(this._oFilterDialog);
            }
            this._oFilterDialog.open();
        },

        onRefreshAlerts: function() {
            this._loadAlerts();
            MessageToast.show(this.getResourceBundle().getText("alerts.refresh.success"));
        },

        onAlertPress: function(oEvent) {
            var oAlert = oEvent.getSource().getBindingContext("alerts").getObject();
            this.oAlertsModel.setProperty("/selectedAlert", oAlert);
            this.byId("alertDetailDialog").open();
        },

        onCloseAlertDetail: function() {
            this.byId("alertDetailDialog").close();
        },

        onAcknowledgeAlert: function(oEvent) {
            var oAlert = oEvent.getSource().getBindingContext("alerts").getObject();
            oAlert.acknowledged = true;
            oAlert.actionsTaken = oAlert.actionsTaken || [];
            oAlert.actionsTaken.push({
                action: "Acknowledged",
                user: "current.user@sap.com",
                timestamp: new Date()
            });
            
            this.oAlertsModel.refresh();
            MessageToast.show(this.getResourceBundle().getText("alerts.acknowledge.success"));
        },

        onResolveAlert: function(oEvent) {
            var oAlert = oEvent.getSource().getBindingContext("alerts").getObject();
            
            MessageBox.confirm(
                this.getResourceBundle().getText("alerts.resolve.confirm"),
                {
                    title: this.getResourceBundle().getText("alerts.resolve.title"),
                    onClose: function(sAction) {
                        if (sAction === MessageBox.Action.OK) {
                            this._resolveAlert(oAlert);
                        }
                    }.bind(this)
                }
            );
        },

        _resolveAlert: function(oAlert) {
            // Remove from active alerts
            var aActiveAlerts = this.oAlertsModel.getProperty("/activeAlerts");
            var iIndex = aActiveAlerts.indexOf(oAlert);
            if (iIndex > -1) {
                aActiveAlerts.splice(iIndex, 1);
            }
            
            // Add to history
            var aHistory = this.oAlertsModel.getProperty("/alertHistory");
            oAlert.status = "resolved";
            oAlert.resolvedBy = "current.user@sap.com";
            oAlert.resolvedAt = new Date();
            oAlert.duration = Math.floor((oAlert.resolvedAt - oAlert.timestamp) / 60000) + " min";
            aHistory.unshift(oAlert);
            
            this.oAlertsModel.setProperty("/activeAlerts", aActiveAlerts);
            this.oAlertsModel.setProperty("/alertHistory", aHistory);
            this._updateStatistics();
            
            MessageToast.show(this.getResourceBundle().getText("alerts.resolve.success"));
        },

        onEscalateAlert: function(oEvent) {
            var oAlert = oEvent.getSource().getBindingContext("alerts").getObject();
            
            // Increase severity
            var aSeverities = ["low", "medium", "high", "critical"];
            var iCurrentIndex = aSeverities.indexOf(oAlert.severity);
            if (iCurrentIndex < aSeverities.length - 1) {
                oAlert.severity = aSeverities[iCurrentIndex + 1];
                oAlert.actionsTaken = oAlert.actionsTaken || [];
                oAlert.actionsTaken.push({
                    action: "Escalated to " + oAlert.severity,
                    user: "current.user@sap.com",
                    timestamp: new Date()
                });
                
                this.oAlertsModel.refresh();
                this._updateStatistics();
                
                // Send escalation notifications
                this._sendEscalationNotifications(oAlert);
                
                MessageToast.show(this.getResourceBundle().getText("alerts.escalate.success"));
            }
        },

        onDismissAlert: function(oEvent) {
            var oAlert = oEvent.getSource().getBindingContext("alerts").getObject();
            
            MessageBox.confirm(
                this.getResourceBundle().getText("alerts.dismiss.confirm"),
                {
                    title: this.getResourceBundle().getText("alerts.dismiss.title"),
                    onClose: function(sAction) {
                        if (sAction === MessageBox.Action.OK) {
                            this._dismissAlert(oAlert);
                        }
                    }.bind(this)
                }
            );
        },

        _dismissAlert: function(oAlert) {
            var aActiveAlerts = this.oAlertsModel.getProperty("/activeAlerts");
            var iIndex = aActiveAlerts.indexOf(oAlert);
            if (iIndex > -1) {
                aActiveAlerts.splice(iIndex, 1);
            }
            
            this.oAlertsModel.setProperty("/activeAlerts", aActiveAlerts);
            this._updateStatistics();
            
            MessageToast.show(this.getResourceBundle().getText("alerts.dismiss.success"));
        },

        onHistoryDateRangeChange: function(oEvent) {
            var oDateRange = oEvent.getSource();
            var oTable = this.byId("alertHistoryTable");
            var oBinding = oTable.getBinding("items");
            
            if (oDateRange.getDateValue() && oDateRange.getSecondDateValue()) {
                oBinding.filter(new Filter("timestamp", FilterOperator.BT,
                    oDateRange.getDateValue(), oDateRange.getSecondDateValue()));
            } else {
                oBinding.filter([]);
            }
        },

        onExportHistory: function() {
            var aColumns = [
                { label: "Timestamp", property: "timestamp", type: EdmType.DateTime },
                { label: "Severity", property: "severity", type: EdmType.String },
                { label: "Title", property: "title", type: EdmType.String },
                { label: "Source", property: "source", type: EdmType.String },
                { label: "Status", property: "status", type: EdmType.String },
                { label: "Resolved By", property: "resolvedBy", type: EdmType.String },
                { label: "Duration", property: "duration", type: EdmType.String }
            ];
            
            var oTable = this.byId("alertHistoryTable");
            var aData = oTable.getBinding("items").getContexts().map(function(oContext) {
                return oContext.getObject();
            });
            
            var oSettings = {
                workbook: { columns: aColumns },
                dataSource: aData,
                fileName: "Alert_History_" + new Date().toISOString().split('T')[0] + ".xlsx",
                worker: true
            };
            
            new Spreadsheet(oSettings).build()
                .then(function() {
                    MessageToast.show(this.getResourceBundle().getText("alerts.export.success"));
                }.bind(this));
        },

        onHistoryItemPress: function(oEvent) {
            var oAlert = oEvent.getSource().getBindingContext("alerts").getObject();
            this.oAlertsModel.setProperty("/selectedAlert", oAlert);
            this.byId("alertDetailDialog").open();
        },

        onCreateAlertRule: function() {
            // Open rule creation dialog
            if (!this._oRuleDialog) {
                this._oRuleDialog = sap.ui.xmlfragment(
                    "a2a.network.fiori.fragment.AlertRuleDialog",
                    this
                );
                this.getView().addDependent(this._oRuleDialog);
            }
            
            // Clear form
            this.oAlertsModel.setProperty("/newRule", {
                name: "",
                description: "",
                type: "Threshold",
                severity: "medium",
                condition: "",
                notifications: []
            });
            
            this._oRuleDialog.open();
        },

        onToggleActiveRules: function(oEvent) {
            var bShowActive = oEvent.getParameter("state");
            var oTable = this.byId("alertRulesTable");
            var oBinding = oTable.getBinding("items");
            
            if (bShowActive) {
                oBinding.filter(new Filter("active", FilterOperator.EQ, true));
            } else {
                oBinding.filter([]);
            }
        },

        onRuleSelectionChange: function(oEvent) {
            var oSelectedItem = oEvent.getParameter("listItem");
            if (oSelectedItem) {
                var oRule = oSelectedItem.getBindingContext("alerts").getObject();
                this.oAlertsModel.setProperty("/selectedRule", oRule);
            }
        },

        onToggleRule: function(oEvent) {
            var bActive = oEvent.getParameter("state");
            var oRule = oEvent.getSource().getBindingContext("alerts").getObject();
            
            if (bActive) {
                MessageToast.show(this.getResourceBundle().getText("alerts.rule.activated", [oRule.name]));
            } else {
                MessageToast.show(this.getResourceBundle().getText("alerts.rule.deactivated", [oRule.name]));
            }
        },

        onEditRule: function(oEvent) {
            var oRule = oEvent.getSource().getBindingContext("alerts").getObject();
            
            // Open edit dialog with rule data
            if (!this._oRuleDialog) {
                this._oRuleDialog = sap.ui.xmlfragment(
                    "a2a.network.fiori.fragment.AlertRuleDialog",
                    this
                );
                this.getView().addDependent(this._oRuleDialog);
            }
            
            this.oAlertsModel.setProperty("/editRule", oRule);
            this._oRuleDialog.open();
        },

        onDeleteRule: function(oEvent) {
            var oRule = oEvent.getSource().getBindingContext("alerts").getObject();
            
            MessageBox.confirm(
                this.getResourceBundle().getText("alerts.rule.delete.confirm", [oRule.name]),
                {
                    title: this.getResourceBundle().getText("alerts.rule.delete.title"),
                    onClose: function(sAction) {
                        if (sAction === MessageBox.Action.OK) {
                            this._deleteRule(oRule);
                        }
                    }.bind(this)
                }
            );
        },

        _deleteRule: function(oRule) {
            var aRules = this.oAlertsModel.getProperty("/alertRules");
            var iIndex = aRules.indexOf(oRule);
            if (iIndex > -1) {
                aRules.splice(iIndex, 1);
            }
            
            this.oAlertsModel.setProperty("/alertRules", aRules);
            MessageToast.show(this.getResourceBundle().getText("alerts.rule.delete.success"));
        },

        onNotificationSettings: function() {
            this.byId("notificationSettingsDialog").open();
        },

        onSaveNotificationSettings: function() {
            // Save settings - in production, persist to backend
            var oSettings = this.oSettingsModel.getData();
            localStorage.setItem("a2a_notification_settings", JSON.stringify(oSettings));
            
            this.byId("notificationSettingsDialog").close();
            MessageToast.show(this.getResourceBundle().getText("alerts.notifications.save.success"));
        },

        onCloseNotificationSettings: function() {
            this.byId("notificationSettingsDialog").close();
        },

        onViewRunbook: function() {
            var oAlert = this.oAlertsModel.getProperty("/selectedAlert");
            
            // In production, open runbook documentation
            MessageToast.show(this.getResourceBundle().getText("alerts.runbook.opening"));
            
            // Simulate opening runbook
            window.open("https://wiki.a2a-network.com/runbooks/" + oAlert.source, "_blank");
        },

        onCloseRealtimeAlert: function() {
            this.oUIModel.setProperty("/showRealtimeAlert", false);
        },

        _loadNotificationSettings: function() {
            var sSavedSettings = localStorage.getItem("a2a_notification_settings");
            if (sSavedSettings) {
                try {
                    var oSettings = JSON.parse(sSavedSettings);
                    this.oSettingsModel.setData(oSettings);
                } catch (e) {
                    console.error("Failed to load notification settings:", e);
                }
            }
        },

        _sendNotifications: function(alert) {
            var oSettings = this.oSettingsModel.getData();
            var severityLevel = ["low", "medium", "high", "critical"].indexOf(alert.severity);
            var thresholdLevel = oSettings.notifications.severityThreshold;
            
            if (severityLevel >= thresholdLevel) {
                // Send notifications based on enabled channels
                if (oSettings.notifications.email.enabled) {
                    this._sendEmailNotification(alert, oSettings.notifications.email.address);
                }
                if (oSettings.notifications.sms.enabled) {
                    this._sendSMSNotification(alert, oSettings.notifications.sms.number);
                }
                if (oSettings.notifications.push.enabled) {
                    this._sendPushNotification(alert);
                }
                if (oSettings.notifications.slack.enabled) {
                    this._sendSlackNotification(alert, oSettings.notifications.slack.webhook);
                }
            }
        },

        _sendEmailNotification: function(alert, email) {
            console.log("Sending email notification to:", email, alert);
            // In production, call backend service
        },

        _sendSMSNotification: function(alert, phone) {
            console.log("Sending SMS notification to:", phone, alert);
            // In production, call backend service
        },

        _sendPushNotification: function(alert) {
            console.log("Sending push notification:", alert);
            // In production, use browser notifications API
            if ("Notification" in window && Notification.permission === "granted") {
                new Notification("A2A Network Alert: " + alert.severity.toUpperCase(), {
                    body: alert.title,
                    icon: "/images/alert-icon.png"
                });
            }
        },

        _sendSlackNotification: function(alert, webhook) {
            console.log("Sending Slack notification to:", webhook, alert);
            // In production, call webhook
        },

        _sendEscalationNotifications: function(alert) {
            // Send special notifications for escalated alerts
            var sMessage = "Alert escalated to " + alert.severity.toUpperCase() + ": " + alert.title;
            MessageToast.show(sMessage);
            
            // In production, notify escalation team
            this._sendNotifications(alert);
        },

        _startAlertMonitoring: function() {
            // Check for new alerts every 30 seconds
            this._monitoringInterval = setInterval(function() {
                // In production, poll backend for new alerts
                var random = Math.random();
                if (random > 0.8) {
                    // Simulate new alert
                    var newAlert = this._generateActiveAlerts()[0];
                    this._handleRealtimeAlert(newAlert);
                }
            }.bind(this), 30000);
        },

        onNavBack: function() {
            BaseController.prototype.onNavBack.apply(this, arguments);
        },

        onExit: function() {
            // Clean up WebSocket connections
            if (WebSocketUtil.isConnected()) {
                WebSocketUtil.unsubscribe([
                    'monitoring.alerts',
                    'system.health',
                    'agent.events',
                    'service.events'
                ]).catch((error) => {
                    console.error("Error unsubscribing from WebSocket topics:", error);
                });
                
                // Remove event listeners
                WebSocketUtil.off('alert.new');
                WebSocketUtil.off('alert.updated');
                WebSocketUtil.off('system.health.changed');
                
                WebSocketUtil.disconnect();
            }
            
            // Clean up old WebSocket connection (legacy)
            if (this._wsConnection) {
                this._wsConnection.close();
            }
            
            if (this._monitoringInterval) {
                clearInterval(this._monitoringInterval);
            }
        }
    });
});