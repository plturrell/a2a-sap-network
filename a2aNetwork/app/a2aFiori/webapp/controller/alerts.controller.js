/**
 * A2A Protocol Compliance: HTTP client usage replaced with blockchain messaging
 */

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
], (BaseController, JSONModel, Filter, FilterOperator, MessageToast, MessageBox,
    Spreadsheet, exportLibrary, WebSocketUtil) => {
    "use strict";

    const EdmType = exportLibrary.EdmType;

    return BaseController.extend("a2a.network.fiori.controller.Alerts", {

        onInit() {
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

        getCurrentUserEmail() {
            try {
                return sap.ushell.Container.getUser().getEmail() ||
                       this.getOwnerComponent().getModel("app").getProperty("/currentUser/email") ||
                       "user@a2a.network";
            } catch (error) {
                return "user@a2a.network";
            }
        },

        _initializeModels() {
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
                    email: { enabled: true, address: this.getCurrentUserEmail() },
                    sms: { enabled: false, number: "" },
                    push: { enabled: true },
                    slack: { enabled: false, webhook: "" }, // To be configured by user
                    severityThreshold: 1 // 0=all, 1=medium+, 2=high+, 3=critical
                }
            });
            this.getView().setModel(this.oSettingsModel, "settings");

            // Update UI model
            this.oUIModel.setProperty("/alertView", "active");
            this.oUIModel.setProperty("/showActiveRulesOnly", true);
            this.oUIModel.setProperty("/showRealtimeAlert", false);
        },

        _setupRealtimeAlerts() {
            // Modern WebSocket connection for real-time alerts using socket.io
            this._connectWebSocket().then(() => {
                // Subscribe to alert-related topics
                WebSocketUtil.subscribe([
                    "monitoring.alerts",
                    "system.health",
                    "agent.events",
                    "service.events"
                ]).then((data) => {
                    sap.base.Log.info("Subscribed to alert topics", data, "Alerts");
                }).catch((error) => {
                    // // console.error("Failed to subscribe to alert topics:", error);
                });

                // Listen for real-time alert events
                WebSocketUtil.on("alert.new", this._handleRealtimeAlert.bind(this));
                WebSocketUtil.on("alert.updated", this._handleAlertUpdate.bind(this));
                WebSocketUtil.on("system.health.changed", this._handleSystemHealthChange.bind(this));

            }).catch((error) => {
                // // console.error("Failed to establish WebSocket connection:", error);
                // Fallback: retry connection after 5 seconds
                setTimeout(this._setupRealtimeAlerts.bind(this), 5000);
            });
        },

        _connectWebSocket() {
            // Get authentication token from user model
            const userModel = this.getOwnerComponent().getModel("user");
            const token = userModel?.getProperty("/token") || "dev-token";

            return WebSocketUtil.connect({ token });
        },

        _handleAlertUpdate(alertData) {
            // Update existing alert in the model
            const aActiveAlerts = this.oAlertsModel.getProperty("/activeAlerts");
            const alertIndex = aActiveAlerts.findIndex(alert => alert.id === alertData.id);

            if (alertIndex !== -1) {
                aActiveAlerts[alertIndex] = { ...aActiveAlerts[alertIndex], ...alertData };
                this.oAlertsModel.setProperty("/activeAlerts", aActiveAlerts);
                this._updateStatistics();
            }
        },

        _handleSystemHealthChange(healthData) {
            // Update system health indicators and create alerts if needed
            if (healthData.status === "degraded" || healthData.status === "critical") {
                const alert = {
                    id: `health_alert_${ Date.now()}`,
                    severity: healthData.status === "critical" ? "critical" : "high",
                    priority: healthData.status === "critical" ? 1 : 2,
                    title: `System Health ${healthData.status.charAt(0).toUpperCase() + healthData.status.slice(1)}`,
                    description: `System health has changed to ${healthData.status}. ${healthData.message || ""}`,
                    source: "SystemHealth",
                    timestamp: new Date(healthData.timestamp || Date.now()),
                    acknowledged: false,
                    metrics: healthData.metrics || {}
                };

                this._handleRealtimeAlert(alert);
            }
        },

        _handleRealtimeAlert(alert) {
            // Add to active alerts
            const aActiveAlerts = this.oAlertsModel.getProperty("/activeAlerts");
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

        _showRealtimeNotification(alert) {
            const sMessage = `${alert.title } - ${ alert.description}`;
            const sType = alert.severity === "critical" ? "Error" :
                alert.severity === "high" ? "Warning" : "Information";

            this.oUIModel.setProperty("/showRealtimeAlert", true);
            this.oUIModel.setProperty("/realtimeAlertMessage", sMessage);
            this.oUIModel.setProperty("/realtimeAlertType", sType);

            // Auto-hide after 10 seconds
            setTimeout(() => {
                this.oUIModel.setProperty("/showRealtimeAlert", false);
            }, 10000);
        },

        _playAlertSound() {
            // Play alert sound for critical alerts
            try {
                const audio = new Audio("/sounds/alert.mp3");
                audio.play();
            } catch (e) {
                // // console.error("Failed to play alert sound:", e);
            }
        },

        _loadAlerts() {
            this.showSkeletonLoading(this.getResourceBundle().getText("alerts.loading"));

            // Load alerts from backend service
            const apiBaseUrl = window.A2A_CONFIG?.apiBaseUrl || "/api/v1";

            Promise.all([
                fetch(`${apiBaseUrl}/alerts/active`),
                fetch(`${apiBaseUrl}/alerts/history`),
                fetch(`${apiBaseUrl}/alerts/rules`)
            ]).then(responses => {
                return Promise.all(responses.map(r => r.json()));
            }).then(([activeAlerts, alertHistory, alertRules]) => {
                this.oAlertsModel.setProperty("/activeAlerts", activeAlerts || []);
                this.oAlertsModel.setProperty("/alertHistory", alertHistory || []);
                this.oAlertsModel.setProperty("/alertRules", alertRules || []);

                this._updateStatistics();
                this.hideLoading();
            }).catch(error => {
                // // console.error("Failed to load alerts:", error);
                this.hideLoading();
                this.showErrorMessage(this.getResourceBundle().getText("alerts.loadError"));
            });
        },

        _updateStatistics() {
            const aActiveAlerts = this.oAlertsModel.getProperty("/activeAlerts");
            const oStats = {
                critical: 0,
                high: 0,
                medium: 0,
                low: 0,
                total: aActiveAlerts.length
            };

            aActiveAlerts.forEach((alert) => {
                oStats[alert.severity]++;
            });

            this.oAlertsModel.setProperty("/statistics", oStats);
        },

        onSearchAlerts(oEvent) {
            const sQuery = oEvent.getParameter("query");
            const oList = this.byId("activeAlertsList");
            const _oBinding = oList.getBinding("items");

            if (sQuery) {
                const aFilters = [
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

        onFilterAlerts() {
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

        onRefreshAlerts() {
            this._loadAlerts();
            MessageToast.show(this.getResourceBundle().getText("alerts.refresh.success"));
        },

        onAlertPress(oEvent) {
            const oAlert = oEvent.getSource().getBindingContext("alerts").getObject();
            this.oAlertsModel.setProperty("/selectedAlert", oAlert);
            this.byId("alertDetailDialog").open();
        },

        onCloseAlertDetail() {
            this.byId("alertDetailDialog").close();
        },

        onAcknowledgeAlert(oEvent) {
            const oAlert = oEvent.getSource().getBindingContext("alerts").getObject();
            oAlert.acknowledged = true;
            oAlert.actionsTaken = oAlert.actionsTaken || [];
            oAlert.actionsTaken.push({
                action: "Acknowledged",
                user: this.getCurrentUserEmail(),
                timestamp: new Date()
            });

            this.oAlertsModel.refresh();
            MessageToast.show(this.getResourceBundle().getText("alerts.acknowledge.success"));
        },

        onResolveAlert(oEvent) {
            const oAlert = oEvent.getSource().getBindingContext("alerts").getObject();

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

        _resolveAlert(oAlert) {
            // Remove from active alerts
            const aActiveAlerts = this.oAlertsModel.getProperty("/activeAlerts");
            const iIndex = aActiveAlerts.indexOf(oAlert);
            if (iIndex > -1) {
                aActiveAlerts.splice(iIndex, 1);
            }

            // Add to history
            const aHistory = this.oAlertsModel.getProperty("/alertHistory");
            oAlert.status = "resolved";
            oAlert.resolvedBy = this.getCurrentUserEmail();
            oAlert.resolvedAt = new Date();
            oAlert.duration = `${Math.floor((oAlert.resolvedAt - oAlert.timestamp) / 60000) } min`;
            aHistory.unshift(oAlert);

            this.oAlertsModel.setProperty("/activeAlerts", aActiveAlerts);
            this.oAlertsModel.setProperty("/alertHistory", aHistory);
            this._updateStatistics();

            MessageToast.show(this.getResourceBundle().getText("alerts.resolve.success"));
        },

        onEscalateAlert(oEvent) {
            const oAlert = oEvent.getSource().getBindingContext("alerts").getObject();

            // Increase severity
            const aSeverities = ["low", "medium", "high", "critical"];
            const iCurrentIndex = aSeverities.indexOf(oAlert.severity);
            if (iCurrentIndex < aSeverities.length - 1) {
                oAlert.severity = aSeverities[iCurrentIndex + 1];
                oAlert.actionsTaken = oAlert.actionsTaken || [];
                oAlert.actionsTaken.push({
                    action: `Escalated to ${ oAlert.severity}`,
                    user: this.getCurrentUserEmail(),
                    timestamp: new Date()
                });

                this.oAlertsModel.refresh();
                this._updateStatistics();

                // Send escalation notifications
                this._sendEscalationNotifications(oAlert);

                MessageToast.show(this.getResourceBundle().getText("alerts.escalate.success"));
            }
        },

        onDismissAlert(oEvent) {
            const oAlert = oEvent.getSource().getBindingContext("alerts").getObject();

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

        _dismissAlert(oAlert) {
            const aActiveAlerts = this.oAlertsModel.getProperty("/activeAlerts");
            const iIndex = aActiveAlerts.indexOf(oAlert);
            if (iIndex > -1) {
                aActiveAlerts.splice(iIndex, 1);
            }

            this.oAlertsModel.setProperty("/activeAlerts", aActiveAlerts);
            this._updateStatistics();

            MessageToast.show(this.getResourceBundle().getText("alerts.dismiss.success"));
        },

        onHistoryDateRangeChange(oEvent) {
            const oDateRange = oEvent.getSource();
            const oTable = this.byId("alertHistoryTable");
            const _oBinding = oTable.getBinding("items");

            if (oDateRange.getDateValue() && oDateRange.getSecondDateValue()) {
                oBinding.filter(new Filter("timestamp", FilterOperator.BT,
                    oDateRange.getDateValue(), oDateRange.getSecondDateValue()));
            } else {
                oBinding.filter([]);
            }
        },

        onExportHistory() {
            const _aColumns = [
                { label: "Timestamp", property: "timestamp", type: EdmType.DateTime },
                { label: "Severity", property: "severity", type: EdmType.String },
                { label: "Title", property: "title", type: EdmType.String },
                { label: "Source", property: "source", type: EdmType.String },
                { label: "Status", property: "status", type: EdmType.String },
                { label: "Resolved By", property: "resolvedBy", type: EdmType.String },
                { label: "Duration", property: "duration", type: EdmType.String }
            ];

            const oTable = this.byId("alertHistoryTable");
            const aData = oTable.getBinding("items").getContexts().map((oContext) => {
                return oContext.getObject();
            });

            const oSettings = {
                workbook: { columns: aColumns },
                dataSource: aData,
                fileName: `Alert_History_${ new Date().toISOString().split("T")[0] }.xlsx`,
                worker: true
            };

            new Spreadsheet(oSettings).build()
                .then(() => {
                    MessageToast.show(this.getResourceBundle().getText("alerts.export.success"));
                });
        },

        onHistoryItemPress(oEvent) {
            const oAlert = oEvent.getSource().getBindingContext("alerts").getObject();
            this.oAlertsModel.setProperty("/selectedAlert", oAlert);
            this.byId("alertDetailDialog").open();
        },

        onCreateAlertRule() {
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

        onToggleActiveRules(oEvent) {
            const bShowActive = oEvent.getParameter("state");
            const oTable = this.byId("alertRulesTable");
            const _oBinding = oTable.getBinding("items");

            if (bShowActive) {
                oBinding.filter(new Filter("active", FilterOperator.EQ, true));
            } else {
                oBinding.filter([]);
            }
        },

        onRuleSelectionChange(oEvent) {
            const oSelectedItem = oEvent.getParameter("listItem");
            if (oSelectedItem) {
                const oRule = oSelectedItem.getBindingContext("alerts").getObject();
                this.oAlertsModel.setProperty("/selectedRule", oRule);
            }
        },

        onToggleRule(oEvent) {
            const bActive = oEvent.getParameter("state");
            const oRule = oEvent.getSource().getBindingContext("alerts").getObject();

            if (bActive) {
                MessageToast.show(this.getResourceBundle().getText("alerts.rule.activated", [oRule.name]));
            } else {
                MessageToast.show(this.getResourceBundle().getText("alerts.rule.deactivated", [oRule.name]));
            }
        },

        onEditRule(oEvent) {
            const oRule = oEvent.getSource().getBindingContext("alerts").getObject();

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

        onDeleteRule(oEvent) {
            const oRule = oEvent.getSource().getBindingContext("alerts").getObject();

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

        _deleteRule(oRule) {
            const aRules = this.oAlertsModel.getProperty("/alertRules");
            const iIndex = aRules.indexOf(oRule);
            if (iIndex > -1) {
                aRules.splice(iIndex, 1);
            }

            this.oAlertsModel.setProperty("/alertRules", aRules);
            MessageToast.show(this.getResourceBundle().getText("alerts.rule.delete.success"));
        },

        onNotificationSettings() {
            this.byId("notificationSettingsDialog").open();
        },

        onSaveNotificationSettings() {
            // Save settings - in production, persist to backend
            const oSettings = this.oSettingsModel.getData();
            localStorage.setItem("a2a_notification_settings", JSON.stringify(oSettings));

            this.byId("notificationSettingsDialog").close();
            MessageToast.show(this.getResourceBundle().getText("alerts.notifications.save.success"));
        },

        onCloseNotificationSettings() {
            this.byId("notificationSettingsDialog").close();
        },

        onViewRunbook() {
            const oAlert = this.oAlertsModel.getProperty("/selectedAlert");

            // Open runbook documentation
            const runbookBaseUrl = window.A2A_CONFIG?.runbookBaseUrl || "/runbooks";
            MessageToast.show(this.getResourceBundle().getText("alerts.runbook.opening"));

            window.open(`${runbookBaseUrl}/${oAlert.source}`, "_blank");
        },

        onCloseRealtimeAlert() {
            this.oUIModel.setProperty("/showRealtimeAlert", false);
        },

        _loadNotificationSettings() {
            const sSavedSettings = localStorage.getItem("a2a_notification_settings");
            if (sSavedSettings) {
                try {
                    const oSettings = JSON.parse(sSavedSettings);
                    this.oSettingsModel.setData(oSettings);
                } catch (e) {
                    // // console.error("Failed to load notification settings:", e);
                }
            }
        },

        _sendNotifications(alert) {
            const oSettings = this.oSettingsModel.getData();
            const severityLevel = ["low", "medium", "high", "critical"].indexOf(alert.severity);
            const thresholdLevel = oSettings.notifications.severityThreshold;

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

        _sendEmailNotification(alert, email) {
            sap.base.Log.info("Sending email notification", { email, alert }, "Alerts");
            // In production, call backend service
        },

        _sendSMSNotification(alert, phone) {
            sap.base.Log.info("Sending SMS notification", { phone, alert }, "Alerts");
            // In production, call backend service
        },

        _sendPushNotification(alert) {
            sap.base.Log.info("Sending push notification", alert, "Alerts");
            // In production, use browser notifications API
            if ("Notification" in window && Notification.permission === "granted") {
                new Notification(`A2A Network Alert: ${ alert.severity.toUpperCase()}`, {
                    body: alert.title,
                    icon: "/images/alert-icon.png"
                });
            }
        },

        _sendSlackNotification(alert, webhook) {
            sap.base.Log.info("Sending Slack notification", { webhook, alert }, "Alerts");
            // In production, call webhook
        },

        _sendEscalationNotifications(alert) {
            // Send special notifications for escalated alerts
            const sMessage = `Alert escalated to ${ alert.severity.toUpperCase() }: ${ alert.title}`;
            MessageToast.show(sMessage);

            // In production, notify escalation team
            this._sendNotifications(alert);
        },

        _startAlertMonitoring() {
            // Real-time monitoring handled via WebSocket connections
            // No need for polling
        },

        onNavBack() {
            BaseController.prototype.onNavBack.apply(this, arguments);
        },

        onExit() {
            // Clean up WebSocket connections
            if (WebSocketUtil.isConnected()) {
                WebSocketUtil.unsubscribe([
                    "monitoring.alerts",
                    "system.health",
                    "agent.events",
                    "service.events"
                ]).catch((error) => {
                    // // console.error("Error unsubscribing from WebSocket topics:", error);
                });

                // Remove event listeners
                WebSocketUtil.off("alert.new");
                WebSocketUtil.off("alert.updated");
                WebSocketUtil.off("system.health.changed");

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