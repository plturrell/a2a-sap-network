sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/m/MessageToast",
    "../model/formatter"
], function(Controller, MessageToast, formatter) {
    "use strict";

    return Controller.extend("a2a.network.fiori.controller.Workflows", {
        formatter: formatter,

        onInit: function() {
            this.getRouter().getRoute("workflows").attachPatternMatched(this._onRouteMatched, this);
        },

        _onRouteMatched: function() {
            this.getView().getModel().refresh();
        },

        onCreateWorkflow: function() {
            MessageToast.show("Create Workflow - Coming Soon");
        },

        onWorkflowSelect: function(oEvent) {
            var oItem = oEvent.getParameter("listItem");
            var oContext = oItem.getBindingContext();
            
            // Update header
            var oHeader = this.byId("workflowHeader");
            oHeader.setBindingContext(oContext);
            
            // Update executions table
            var oTable = this.byId("executionsTable");
            oTable.bindItems({
                path: oContext.getPath() + "/executions",
                template: oTable.getItems()[0].clone()
            });
        },

        onExecuteWorkflow: function() {
            MessageToast.show("Execute Workflow");
        },

        onEditWorkflow: function() {
            MessageToast.show("Edit Workflow");
        },

        onDeleteWorkflow: function() {
            MessageToast.show("Delete Workflow");
        },

        getRouter: function() {
            return this.getOwnerComponent().getRouter();
        }
    });
});