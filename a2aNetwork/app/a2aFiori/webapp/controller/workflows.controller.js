sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/m/MessageToast",
    "../model/formatter"
], function(Controller, MessageToast, formatter) {
    "use strict";

    return Controller.extend("a2a.network.fiori.controller.Workflows", {
        formatter,

        onInit() {
            this.getRouter().getRoute("workflows").attachPatternMatched(this._onRouteMatched, this);
        },

        _onRouteMatched() {
            this.getView().getModel().refresh();
        },

        onCreateWorkflow() {
            MessageToast.show("Create Workflow - Coming Soon");
        },

        onWorkflowSelect(oEvent) {
            const oItem = oEvent.getParameter("listItem");
            const _oContext = oItem.getBindingContext();

            // Update header
            const oHeader = this.byId("workflowHeader");
            oHeader.setBindingContext(oContext);

            // Update executions table
            const oTable = this.byId("executionsTable");
            oTable.bindItems({
                path: `${oContext.getPath() }/executions`,
                template: oTable.getItems()[0].clone()
            });
        },

        onExecuteWorkflow() {
            MessageToast.show("Execute Workflow");
        },

        onEditWorkflow() {
            MessageToast.show("Edit Workflow");
        },

        onDeleteWorkflow() {
            MessageToast.show("Delete Workflow");
        },

        getRouter() {
            return this.getOwnerComponent().getRouter();
        }
    });
});