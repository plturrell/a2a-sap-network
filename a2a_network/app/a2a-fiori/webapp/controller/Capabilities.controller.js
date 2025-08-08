sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/m/MessageToast",
    "../model/formatter"
], function(Controller, MessageToast, formatter) {
    "use strict";

    return Controller.extend("a2a.network.fiori.controller.Capabilities", {
        formatter: formatter,

        onInit: function() {
            this.getRouter().getRoute("capabilities").attachPatternMatched(this._onRouteMatched, this);
        },

        _onRouteMatched: function() {
            this.getView().getModel().refresh();
        },

        onRegisterCapability: function() {
            MessageToast.show("Register Capability - Coming Soon");
        },

        onTabSelect: function(oEvent) {
            var sKey = oEvent.getParameter("key");
            // Filter capabilities by category
            MessageToast.show("Filter by: " + sKey);
        },

        onSearch: function(oEvent) {
            var sQuery = oEvent.getParameter("query");
            MessageToast.show("Searching for: " + sQuery);
        },

        onGroup: function() {
            MessageToast.show("Grouping capabilities");
        },

        onFilter: function() {
            MessageToast.show("Advanced filters");
        },

        getRouter: function() {
            return this.getOwnerComponent().getRouter();
        }
    });
});