sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/core/routing/History",
    "sap/m/MessageToast",
    "../model/formatter"
], function(Controller, History, MessageToast, formatter) {
    "use strict";

    return Controller.extend("a2a.network.fiori.controller.AgentDetail", {
        formatter: formatter,

        onInit: function() {
            this.getRouter().getRoute("agentDetail").attachPatternMatched(this._onRouteMatched, this);
        },

        _onRouteMatched: function(oEvent) {
            var sAgentId = oEvent.getParameter("arguments").agentId;
            this.getView().bindElement({
                path: "/Agents('" + sAgentId + "')",
                parameters: {
                    expand: "capabilities,services,performance"
                }
            });
        },

        onEdit: function() {
            MessageToast.show("Edit mode - Coming Soon");
        },

        onSync: function() {
            var oContext = this.getView().getBindingContext();
            if (!oContext) return;
            
            var sAgentId = oContext.getProperty("ID");
            var oModel = this.getView().getModel();
            
            oModel.callFunction("/Agents('" + sAgentId + "')/registerOnBlockchain", {
                method: "POST",
                success: function() {
                    MessageToast.show("Agent synced with blockchain");
                },
                error: function() {
                    MessageToast.show("Sync failed");
                }
            });
        },

        onNavBack: function() {
            var oHistory = History.getInstance();
            var sPreviousHash = oHistory.getPreviousHash();

            if (sPreviousHash !== undefined) {
                window.history.go(-1);
            } else {
                this.getRouter().navTo("agents");
            }
        },

        getRouter: function() {
            return this.getOwnerComponent().getRouter();
        }
    });
});