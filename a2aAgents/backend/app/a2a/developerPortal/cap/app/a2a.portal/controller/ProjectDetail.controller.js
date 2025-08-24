sap.ui.define([
  'sap/ui/core/mvc/Controller',
  'sap/ui/model/json/JSONModel',
  'sap/m/MessageToast',
  'sap/ui/core/format/DateFormat'
], (Controller, JSONModel, MessageToast, DateFormat) => {
  'use strict';

  return Controller.extend('a2a.portal.controller.ProjectDetail', {

    onInit: function () {
      // Get router
      const oRouter = this.getOwnerComponent().getRouter();
      oRouter.getRoute('projectDetail').attachPatternMatched(this._onPatternMatched, this);
    },

    _onPatternMatched: function (oEvent) {
      const sProjectId = oEvent.getParameter('args').projectId;
      this._loadProjectDetails(sProjectId);
    },

    _loadProjectDetails: function (sProjectId) {
      const oView = this.getView();
      oView.setBusy(true);

      jQuery.ajax({
        url: `/api/projects/${  sProjectId}`,
        method: 'GET',
        success: function (data) {
          oView.setModel(new JSONModel(data));
          oView.setBusy(false);
        }.bind(this),
        error: function (xhr, status, error) {
          MessageToast.show(`Failed to load project details: ${  error}`);
          oView.setBusy(false);
        }.bind(this)
      });
    },

    onNavBack: function () {
      this.getOwnerComponent().getRouter().navTo('projects');
    },

    onEditProject: function () {
      MessageToast.show('Edit project functionality coming soon');
    },

    onDeployProject: function () {
      MessageToast.show('Deploy project functionality coming soon');
    },

    onMoreActions: function (_oEvent) {
      MessageToast.show('More actions menu coming soon');
    },

    onTabSelect: function (oEvent) {
      const sKey = oEvent.getParameter('key');
      // eslint-disable-next-line no-console
      // eslint-disable-next-line no-console
      console.log('Tab selected:', sKey);
    },

    onCreateAgent: function () {
      const oRouter = this.getOwnerComponent().getRouter();
      const sProjectId = this.getView().getModel().getProperty('/project_id');
      oRouter.navTo('agentBuilder', {
        projectId: sProjectId
      });
    },

    onEditAgent: function (oEvent) {
      const oContext = oEvent.getSource().getBindingContext();
      const sAgentId = oContext.getProperty('id');
      MessageToast.show(`Edit agent: ${  sAgentId}`);
    },

    onDeleteAgent: function (oEvent) {
      const oContext = oEvent.getSource().getBindingContext();
      const sAgentName = oContext.getProperty('name');
      MessageToast.show(`Delete agent: ${  sAgentName}`);
    },

    onCreateWorkflow: function () {
      const oRouter = this.getOwnerComponent().getRouter();
      const sProjectId = this.getView().getModel().getProperty('/project_id');
      oRouter.navTo('bpmnDesigner', {
        projectId: sProjectId
      });
    },

    onEditWorkflow: function (oEvent) {
      const oContext = oEvent.getSource().getBindingContext();
      const sWorkflowId = oContext.getProperty('id');
      MessageToast.show(`Edit workflow: ${  sWorkflowId}`);
    },

    onRunWorkflow: function (oEvent) {
      const oContext = oEvent.getSource().getBindingContext();
      const sWorkflowName = oContext.getProperty('name');
      MessageToast.show(`Run workflow: ${  sWorkflowName}`);
    },

    // Formatters
    formatDate: function (sDate) {
      if (!sDate) {
        return '';
      }
            
      const oDateFormat = DateFormat.getDateTimeInstance({
        style: 'medium'
      });
            
      return oDateFormat.format(new Date(sDate));
    },

    formatStatusState: function (sStatus) {
      switch (sStatus) {
      case 'active': return 'Success';
      case 'inactive': return 'Warning';
      case 'error': return 'Error';
      default: return 'None';
      }
    }
  });
});