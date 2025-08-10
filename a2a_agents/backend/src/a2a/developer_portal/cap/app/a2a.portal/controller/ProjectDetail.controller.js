sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast",
    "sap/ui/core/format/DateFormat"
], function (Controller, JSONModel, MessageToast, DateFormat) {
    "use strict";

    return Controller.extend("a2a.portal.controller.ProjectDetail", {

        onInit: function () {
            // Get router
            var oRouter = this.getOwnerComponent().getRouter();
            oRouter.getRoute("projectDetail").attachPatternMatched(this._onPatternMatched, this);
        },

        _onPatternMatched: function (oEvent) {
            var sProjectId = oEvent.getParameter("arguments").projectId;
            this._loadProjectDetails(sProjectId);
        },

        _loadProjectDetails: function (sProjectId) {
            var oView = this.getView();
            oView.setBusy(true);

            jQuery.ajax({
                url: "/api/projects/" + sProjectId,
                method: "GET",
                success: function (data) {
                    oView.setModel(new JSONModel(data));
                    oView.setBusy(false);
                }.bind(this),
                error: function (xhr, status, error) {
                    MessageToast.show("Failed to load project details: " + error);
                    oView.setBusy(false);
                }.bind(this)
            });
        },

        onNavBack: function () {
            this.getOwnerComponent().getRouter().navTo("projects");
        },

        onEditProject: function () {
            MessageToast.show("Edit project functionality coming soon");
        },

        onDeployProject: function () {
            MessageToast.show("Deploy project functionality coming soon");
        },

        onMoreActions: function (oEvent) {
            MessageToast.show("More actions menu coming soon");
        },

        onTabSelect: function (oEvent) {
            var sKey = oEvent.getParameter("key");
            console.log("Tab selected:", sKey);
        },

        onCreateAgent: function () {
            var oRouter = this.getOwnerComponent().getRouter();
            var sProjectId = this.getView().getModel().getProperty("/project_id");
            oRouter.navTo("agentBuilder", {
                projectId: sProjectId
            });
        },

        onEditAgent: function (oEvent) {
            var oContext = oEvent.getSource().getBindingContext();
            var sAgentId = oContext.getProperty("id");
            MessageToast.show("Edit agent: " + sAgentId);
        },

        onDeleteAgent: function (oEvent) {
            var oContext = oEvent.getSource().getBindingContext();
            var sAgentName = oContext.getProperty("name");
            MessageToast.show("Delete agent: " + sAgentName);
        },

        onCreateWorkflow: function () {
            var oRouter = this.getOwnerComponent().getRouter();
            var sProjectId = this.getView().getModel().getProperty("/project_id");
            oRouter.navTo("bpmnDesigner", {
                projectId: sProjectId
            });
        },

        onEditWorkflow: function (oEvent) {
            var oContext = oEvent.getSource().getBindingContext();
            var sWorkflowId = oContext.getProperty("id");
            MessageToast.show("Edit workflow: " + sWorkflowId);
        },

        onRunWorkflow: function (oEvent) {
            var oContext = oEvent.getSource().getBindingContext();
            var sWorkflowName = oContext.getProperty("name");
            MessageToast.show("Run workflow: " + sWorkflowName);
        },

        // Formatters
        formatDate: function (sDate) {
            if (!sDate) {
                return "";
            }
            
            var oDateFormat = DateFormat.getDateTimeInstance({
                style: "medium"
            });
            
            return oDateFormat.format(new Date(sDate));
        },

        formatStatusState: function (sStatus) {
            switch (sStatus) {
                case "active": return "Success";
                case "inactive": return "Warning";
                case "error": return "Error";
                default: return "None";
            }
        }
    });
});