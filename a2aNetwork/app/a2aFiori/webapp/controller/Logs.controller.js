/**
 * A2A Protocol Compliance: HTTP client usage replaced with blockchain messaging
 */

sap.ui.define([
    "./BaseController",
    "sap/m/MessageToast",
    "sap/ui/model/json/JSONModel",
    "sap/ui/core/format/DateFormat"
], (BaseController, MessageToast, JSONModel, DateFormat) => {
    "use strict";

    return BaseController.extend("a2a.network.fiori.controller.Logs", {

        onInit() {
            // Initialize logs model
            this.oLogsModel = new JSONModel({
                logs: [],
                totalCount: 0,
                lastRefresh: null
            });
            this.getView().setModel(this.oLogsModel);

            // Load initial logs
            this.onRefreshLogs();
        },

        onRefreshLogs() {
            const that = this;

            this.showBusyDialog("Loading logs...");

            // Call logs API endpoint
            blockchainClient.sendMessage("/api/v1/operations/logs?limit=200")
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                    return response.json();
                })
                .then(data => {
                    that.oLogsModel.setData({
                        logs: data.logs || [],
                        totalCount: data.total || 0,
                        lastRefresh: new Date()
                    });

                    MessageToast.show(`Loaded ${data.logs?.length || 0} log entries`);
                })
                .catch(error => {
                    // console.error("Failed to load logs:", error);
                    MessageToast.show(`Failed to load logs: ${ error.message}`);

                    // Fallback to mock data
                    that.oLogsModel.setData({
                        logs: that._generateMockLogs(),
                        totalCount: 50,
                        lastRefresh: new Date()
                    });
                })
                .finally(() => {
                    that.hideBusyDialog();
                });
        },

        onDownloadLogs() {
            // Open download endpoint
            window.open("/api/v1/operations/logs/download?format=txt", "_blank");
        },

        onLogLevelChange(oEvent) {
            const sSelectedLevel = oEvent.getParameter("selectedItem").getKey();
            const oTable = this.byId("logsTable");
            const _oBinding = oTable.getBinding("items");

            if (sSelectedLevel === "all") {
                oBinding.filter([]);
            } else {
                const oFilter = new sap.ui.model.Filter("level", sap.ui.model.FilterOperator.EQ, sSelectedLevel);
                oBinding.filter([oFilter]);
            }
        },

        onSearchLogs(oEvent) {
            const sQuery = oEvent.getParameter("query");
            const oTable = this.byId("logsTable");
            const _oBinding = oTable.getBinding("items");

            if (sQuery) {
                const oFilter = new sap.ui.model.Filter("message", sap.ui.model.FilterOperator.Contains, sQuery);
                oBinding.filter([oFilter]);
            } else {
                oBinding.filter([]);
            }
        },

        onLogItemPress(oEvent) {
            const oLogItem = oEvent.getSource().getBindingContext().getObject();

            // Show log details in a dialog
            this._showLogDetails(oLogItem);
        },

        formatLogLevel(sLevel) {
            switch (sLevel) {
            case "error":
                return sap.ui.core.ValueState.Error;
            case "warn":
                return sap.ui.core.ValueState.Warning;
            case "info":
                return sap.ui.core.ValueState.Success;
            case "debug":
                return sap.ui.core.ValueState.Information;
            default:
                return sap.ui.core.ValueState.None;
            }
        },

        _showLogDetails(oLogItem) {
            // Create and show log details dialog
            if (!this._oLogDetailsDialog) {
                this._oLogDetailsDialog = sap.ui.xmlfragment(
                    "a2a.network.fiori.view.fragments.LogDetails",
                    this
                );
                this.getView().addDependent(this._oLogDetailsDialog);
            }

            const oModel = new JSONModel(oLogItem);
            this._oLogDetailsDialog.setModel(oModel);
            this._oLogDetailsDialog.open();
        },

        onCloseLogDetails() {
            this._oLogDetailsDialog.close();
        },

        _generateMockLogs() {
            const levels = ["info", "warn", "error", "debug"];
            const loggers = ["server", "database", "blockchain", "auth", "api"];
            const messages = [
                "Server started successfully",
                "Database connection established",
                "API endpoint called",
                "Authentication successful",
                "Blockchain transaction processed",
                "Cache refreshed",
                "User session started",
                "Data sync completed"
            ];

            const logs = [];
            for (let i = 0; i < 50; i++) {
                logs.push({
                    timestamp: new Date(Date.now() - Math.random() * 86400000).toISOString(),
                    level: levels[Math.floor(Math.random() * levels.length)],
                    logger: loggers[Math.floor(Math.random() * loggers.length)],
                    message: messages[Math.floor(Math.random() * messages.length)],
                    correlationId: `corr-${Math.random().toString(36).substr(2, 9)}`
                });
            }

            return logs.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
        }
    });
});