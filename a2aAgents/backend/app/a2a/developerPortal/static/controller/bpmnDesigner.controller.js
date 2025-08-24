sap.ui.define([
  'sap/ui/core/mvc/Controller',
  'sap/ui/model/json/JSONModel',
  'sap/m/MessageToast',
  'sap/m/MessageBox',
  'sap/ui/core/format/DateFormat'
], (Controller, JSONModel, MessageToast, MessageBox, DateFormat) => {
  'use strict';
  /* global  */

  return Controller.extend('a2a.portal.controller.BPMNDesigner', {

    onInit: function () {
      // Initialize view model
      const oViewModel = new JSONModel({
        projectName: '',
        workflowName: 'New Workflow',
        selectedElement: null,
        lastSaved: null,
        gridEnabled: true,
        zoomLevel: 100
      });
      this.getView().setModel(oViewModel);

      // Initialize workflow model
      this._workflowModel = {
        elements: [],
        connections: []
      };

      // Get router
      const oRouter = this.getOwnerComponent().getRouter();
      oRouter.getRoute('bpmnDesigner').attachPatternMatched(this._onPatternMatched, this);
    },

    _onPatternMatched: function (oEvent) {
      const sProjectId = oEvent.getParameter('args').projectId;
      this._projectId = sProjectId;
      this._loadProjectInfo(sProjectId);
    },

    _loadProjectInfo: function (sProjectId) {
      jQuery.ajax({
        url: `/api/projects/${  sProjectId}`,
        method: 'GET',
        success: function (data) {
          this.getView().getModel().setProperty('/projectName', data.name);
        }.bind(this),
        error: function (xhr, status, error) {
          MessageToast.show(`Failed to load project info: ${  error}`);
        }.bind(this)
      });
    },

    onNavToProjects: function () {
      this.getOwnerComponent().getRouter().navTo('projects');
    },

    onNavToProject: function () {
      this.getOwnerComponent().getRouter().navTo('projectDetail', {
        projectId: this._projectId
      });
    },

    // Component Palette Actions
    onAddServiceTask: function () {
      this._addElement('ServiceTask', 'Service Task', 'sap-icon://activity-2');
    },

    onAddUserTask: function () {
      this._addElement('UserTask', 'User Task', 'sap-icon://user-settings');
    },

    onAddScriptTask: function () {
      this._addElement('ScriptTask', 'Script Task', 'sap-icon://syntax');
    },

    onAddCallActivity: function () {
      this._addElement('CallActivity', 'Call Activity', 'sap-icon://call');
    },

    onAddExclusiveGateway: function () {
      this._addElement('ExclusiveGateway', 'Exclusive Gateway', 'sap-icon://decision');
    },

    onAddParallelGateway: function () {
      this._addElement('ParallelGateway', 'Parallel Gateway', 'sap-icon://split');
    },

    onAddEventGateway: function () {
      this._addElement('EventGateway', 'Event Gateway', 'sap-icon://flag');
    },

    onAddStartEvent: function () {
      this._addElement('StartEvent', 'Start', 'sap-icon://begin');
    },

    onAddEndEvent: function () {
      this._addElement('EndEvent', 'End', 'sap-icon://stop');
    },

    onAddTimerEvent: function () {
      this._addElement('TimerEvent', 'Timer', 'sap-icon://history');
    },

    onAddMessageEvent: function () {
      this._addElement('MessageEvent', 'Message', 'sap-icon://email');
    },

    _addElement: function (sType, sName, sIcon) {
      const oElement = {
        id: `${sType  }_${  Date.now()}`,
        type: sType,
        name: sName,
        icon: sIcon,
        x: 100 + (this._workflowModel.elements.length * 50),
        y: 100 + (this._workflowModel.elements.length * 30),
        width: 100,
        height: 80
      };

      this._workflowModel.elements.push(oElement);
      MessageToast.show(`${sName  } added to canvas`);
            
      // In a real implementation, this would update the canvas
      this._refreshCanvas();
    },

    _refreshCanvas: function () {
      // This would refresh the BPMN canvas with the current model
      // eslint-disable-next-line no-console
      // eslint-disable-next-line no-console
      console.log('Refreshing canvas with', this._workflowModel.elements.length, 'elements');
    },

    // Toolbar Actions
    onUndo: function () {
      MessageToast.show('Undo - coming soon');
    },

    onRedo: function () {
      MessageToast.show('Redo - coming soon');
    },

    onZoomIn: function () {
      const oModel = this.getView().getModel();
      const iZoom = oModel.getProperty('/zoomLevel');
      oModel.setProperty('/zoomLevel', Math.min(iZoom + 10, 200));
      MessageToast.show(`Zoom: ${  oModel.getProperty('/zoomLevel')  }%`);
    },

    onZoomOut: function () {
      const oModel = this.getView().getModel();
      const iZoom = oModel.getProperty('/zoomLevel');
      oModel.setProperty('/zoomLevel', Math.max(iZoom - 10, 50));
      MessageToast.show(`Zoom: ${  oModel.getProperty('/zoomLevel')  }%`);
    },

    onFitToScreen: function () {
      this.getView().getModel().setProperty('/zoomLevel', 100);
      MessageToast.show('Fit to screen');
    },

    onToggleGrid: function () {
      const oModel = this.getView().getModel();
      const bGrid = !oModel.getProperty('/gridEnabled');
      oModel.setProperty('/gridEnabled', bGrid);
      MessageToast.show(`Grid ${  bGrid ? 'enabled' : 'disabled'}`);
    },

    onAutoLayout: function () {
      MessageToast.show('Auto layout - coming soon');
    },

    // Main Actions
    onSaveWorkflow: function () {
      this._saveWorkflow(false);
    },

    onValidateWorkflow: function () {
      MessageToast.show('Validating workflow...');
            
      // Simple validation
      const bHasStart = this._workflowModel.elements.some((el) => {
        return el.type === 'StartEvent';
      });
      const bHasEnd = this._workflowModel.elements.some((el) => {
        return el.type === 'EndEvent';
      });

      if (!bHasStart || !bHasEnd) {
        MessageBox.warning('Workflow must have at least one Start and one End event');
      } else {
        MessageToast.show('Workflow is valid');
      }
    },

    onSimulateWorkflow: function () {
      MessageToast.show('Workflow simulation - coming soon');
    },

    onExportBPMN: function () {
      const sBPMN = this._generateBPMN();
            
      const element = document.createElement('a');
      element.setAttribute('href', `data:text/xml;charset=utf-8,${  encodeURIComponent(sBPMN)}`);
      element.setAttribute('download', 'workflow.bpmn');
      element.style.display = 'none';
      document.body.appendChild(element);
      element.click();
      document.body.removeChild(element);
            
      MessageToast.show('BPMN exported');
    },

    onImportBPMN: function () {
      MessageToast.show('Import BPMN - coming soon');
    },

    onGenerateCode: function () {
      const oRouter = this.getOwnerComponent().getRouter();
      oRouter.navTo('codeEditor', {
        projectId: this._projectId,
        filePath: 'generated_workflow.py'
      });
    },

    onDeleteElement: function () {
      const oModel = this.getView().getModel();
      const oSelected = oModel.getProperty('/selectedElement');
            
      if (oSelected) {
        MessageBox.confirm(`Delete element '${  oSelected.name  }'?`, {
          onClose: function (sAction) {
            if (sAction === MessageBox.Action.OK) {
              // Remove element from model
              const iIndex = this._workflowModel.elements.findIndex((el) => {
                return el.id === oSelected.id;
              });
              if (iIndex > -1) {
                this._workflowModel.elements.splice(iIndex, 1);
                oModel.setProperty('/selectedElement', null);
                this._refreshCanvas();
                MessageToast.show('Element deleted');
              }
            }
          }.bind(this)
        });
      }
    },

    _generateBPMN: function () {
      // Simple BPMN generation
      return `<?xml version="1.0" encoding="UTF-8"?>
<bpmn:definitions xmlns:bpmn="http://www.omg.org/spec/BPMN/20100524/MODEL" 
                  xmlns:bpmndi="http://www.omg.org/spec/BPMN/20100524/DI"
                  id="Definitions_1" targetNamespace="http://a2a.example.com">
    <bpmn:process id="Process_1" isExecutable="true">
        ${this._workflowModel.elements.map((el) => {
    return `<bpmn:${el.type.charAt(0).toLowerCase() + el.type.slice(1)} id="${el.id}" name="${el.name}"/>`;
  }).join('\n        ')}
    </bpmn:process>
</bpmn:definitions>`;
    },

    _saveWorkflow: function (bDraft) {
      const oModel = this.getView().getModel();
      const sWorkflowName = oModel.getProperty('/workflowName');
            
      const oWorkflowData = {
        name: sWorkflowName,
        project_id: this._projectId,
        definition: this._workflowModel,
        draft: bDraft
      };

      jQuery.ajax({
        url: `/api/projects/${  this._projectId  }/workflows`,
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(oWorkflowData),
        success: function (_data) {
          oModel.setProperty('/lastSaved', new Date());
          MessageToast.show('Workflow saved successfully');
        }.bind(this),
        error: function (xhr, status, error) {
          MessageBox.error(`Failed to save workflow: ${  error}`);
        }.bind(this)
      });
    },

    onSaveAsDraft: function () {
      this._saveWorkflow(true);
    },

    onSaveAndActivate: function () {
      this._saveWorkflow(false);
      MessageToast.show('Workflow activated');
    },

    onCancel: function () {
      MessageBox.confirm('Are you sure you want to cancel? Any unsaved changes will be lost.', {
        onClose: function (sAction) {
          if (sAction === MessageBox.Action.OK) {
            this.onNavToProject();
          }
        }.bind(this)
      });
    },

    // Formatter
    formatDateTime: function (oDate) {
      if (!oDate) {
        return 'Never';
      }
            
      const oDateFormat = DateFormat.getDateTimeInstance({
        style: 'medium'
      });
            
      return oDateFormat.format(oDate);
    }
  });
});