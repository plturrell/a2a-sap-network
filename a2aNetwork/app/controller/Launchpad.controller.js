/* global sap */
sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/ui/core/Fragment",
    "sap/m/MessageToast"
], function (Controller, JSONModel, Fragment, MessageToast) {
    "use strict";

    return Controller.extend("a2a.network.launchpad.controller.Launchpad", {
        _intervals: [],


        onInit: function () {
            this.getView().setModel(new JSONModel({ items: [] }), "notifications");
            const oI18nModel = this.getOwnerComponent().getModel("i18n");
            const oResourceBundle = oI18nModel.getResourceBundle();

            const oModel = new JSONModel({
                tiles: [
                    { header: oResourceBundle.getText("agentManagementTileHeader"), subheader: oResourceBundle.getText("agentManagementTileSubheader"), icon: "sap-icon://collaborate", value: 0, info: "agentCount" },
                    { header: oResourceBundle.getText("serviceMarketplaceTileHeader"), subheader: oResourceBundle.getText("serviceMarketplaceTileSubheader"), icon: "sap-icon://sales-order", value: 0, info: "services" },
                    { header: oResourceBundle.getText("workflowDesignerTileHeader"), subheader: oResourceBundle.getText("workflowDesignerTileSubheader"), icon: "sap-icon://workflow-tasks", value: 0, info: "workflows" },
                    { header: oResourceBundle.getText("networkAnalyticsTileHeader"), subheader: oResourceBundle.getText("networkAnalyticsTileSubheader"), icon: "sap-icon://business-objects-experience", value: 0, info: "performance" },
                    { header: oResourceBundle.getText("notificationCenterTileHeader"), subheader: oResourceBundle.getText("notificationCenterTileSubheader"), icon: "sap-icon://bell", value: 0, info: "notifications" },
                    { header: oResourceBundle.getText("securityAuditTileHeader"), subheader: oResourceBundle.getText("securityAuditTileSubheader"), icon: "sap-icon://shield", value: 0, info: "security" }
                ]
            });
            this.getView().setModel(oModel, "launchpad");

            this._fetchAndSetTileData();
            this._intervals.push(setInterval(this._fetchAndSetTileData.bind(this), 30000);
        },

        _fetchAndSetTileData: function () {
            fetch('/api/v1/Agents?id=agent_visualization')
                .then(response => response.ok ? response.json() : Promise.reject('Network response was not ok'))
                .then(data => this._updateModelWithData(data))
                .catch(() => {
                    const fallbackData = { agentCount: 9, services: 0, workflows: 0, performance: 85, notifications: 3, security: 0 };
                    this._updateModelWithData(fallbackData);
                });
        },

        _updateModelWithData: function(data) {
            const oModel = this.getView().getModel("launchpad");
            const aTiles = oModel.getProperty("/tiles");
            aTiles.forEach(oTile => {
                oTile.value = data[oTile.info] || 0;
            });
            oModel.setProperty("/tiles", aTiles);
        },

        onOpenPersonalization: function () {
            if (!this._oPersonalizationDialog) {
                this._oPersonalizationDialog = Fragment.load({
                    name: "a2a.network.launchpad.view.Personalization",
                    controller: this
                }).then(function (oDialog) {
                    this.getView().addDependent(oDialog);
                    return oDialog;
                }.bind(this));
            }

            this._oPersonalizationDialog.then(function(oDialog) {
                oDialog.open();
            });
        },

        onOpenNotifications: function (oEvent) {
            // Sample dynamic notifications
            const oNotificationsModel = this.getView().getModel("notifications");
            oNotificationsModel.setData({
                items: [
                    { title: "System Update", description: "A new system update is available.", icon: "sap-icon://message-information" },
                    { title: "Agent Offline", description: "Agent 'Alpha-7' has gone offline.", icon: "sap-icon://message-warning" },
                    { title: "High-priority Alert", description: "Unusual network activity detected.", icon: "sap-icon://message-error" }
                ]
            });

            if (!this._oNotificationsPopover) {
                this._oNotificationsPopover = Fragment.load({
                    name: "a2a.network.launchpad.view.NotificationCenter",
                    controller: this
                }).then(function (oPopover) {
                    this.getView().addDependent(oPopover);
                    return oPopover;
                }.bind(this));
            }
            this._oNotificationsPopover.then(function (oPopover) {
                oPopover.openBy(oEvent.getSource());
            });
        },

        onClosePersonalization: function () {
            this._oPersonalizationDialog.close();
        },

        onThemeChange: function(oEvent) {
            const sTheme = oEvent.getParameter("selectedItem").getKey();
            sap.ui.getCore().applyTheme(sTheme);
        },

        onRefresh: function () {
            this._fetchAndSetTileData();
            MessageToast.show(this.getOwnerComponent().getModel("i18n").getResourceBundle().getText("refreshSuccessToast"));
        },

        onOpenAnalytics: function () {
            const oRouter = this.getOwnerComponent().getRouter();
            oRouter.navTo("Analytics");
        }
    });

        
        onExit: function() {
            // Clean up intervals to prevent memory leaks
            if (this._intervals) {
                this._intervals.forEach(function(intervalId) {
                    clearInterval(intervalId);
                });
                this._intervals = [];
            }
        }
    });
