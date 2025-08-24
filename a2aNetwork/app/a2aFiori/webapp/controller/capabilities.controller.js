sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/m/MessageToast",
    "../model/formatter"
], (Controller, MessageToast, formatter) => {
    "use strict";

    return Controller.extend("a2a.network.fiori.controller.Capabilities", {
        formatter,

        onInit() {
            this.getRouter().getRoute("capabilities").attachPatternMatched(this._onRouteMatched, this);
        },

        _onRouteMatched() {
            this.getView().getModel().refresh();
        },

        onRegisterCapability() {
            MessageToast.show("Register Capability - Coming Soon");
        },

        onTabSelect(oEvent) {
            const sKey = oEvent.getParameter("key");
            // Filter capabilities by category
            MessageToast.show(`Filter by: ${ sKey}`);
        },

        onSearch(oEvent) {
            const sQuery = oEvent.getParameter("query");
            MessageToast.show(`Searching for: ${ sQuery}`);
        },

        onGroup() {
            MessageToast.show("Grouping capabilities");
        },

        onFilter() {
            MessageToast.show("Advanced filters");
        },

        getRouter() {
            return this.getOwnerComponent().getRouter();
        }
    });
});