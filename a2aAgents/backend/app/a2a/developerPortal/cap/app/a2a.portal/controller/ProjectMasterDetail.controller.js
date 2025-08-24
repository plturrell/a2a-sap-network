sap.ui.define([
  'sap/ui/core/mvc/Controller',
  'sap/ui/model/json/JSONModel',
  'sap/ui/model/Filter',
  'sap/ui/model/FilterOperator',
  'sap/f/library',
  'sap/m/MessageToast',
  'sap/m/MessageBox'
], (Controller, JSONModel, Filter, FilterOperator, fioriLibrary, MessageToast, MessageBox) => {
  'use strict';

  const LayoutType = fioriLibrary.LayoutType;

  return Controller.extend('com.sap.a2a.developerportal.controller.ProjectMasterDetail', {

    onInit: function () {
      this.oRouter = this.getOwnerComponent().getRouter();
      this.oModel = this.getOwnerComponent().getModel();
            
      // Initialize layout model
      this.oLayoutModel = new JSONModel({
        layout: LayoutType.OneColumn,
        actionButtonsInfo: {
          midColumn: {
            fullScreen: false
          }
        }
      });
      this.getView().setModel(this.oLayoutModel);
            
      // Initialize master model
      this.oMasterModel = new JSONModel({
        projects: []
      });
      this.getView().setModel(this.oMasterModel, 'masterModel');
            
      // Initialize detail model
      this.oDetailModel = new JSONModel({});
      this.getView().setModel(this.oDetailModel, 'detailModel');
            
      // Load initial data
      this._loadProjects();
            
      // Register for route matched
      this.oRouter.getRoute('projectMasterDetail').attachPatternMatched(this._onRouteMatched, this);
    },

    _onRouteMatched: function (oEvent) {
      const sProjectId = oEvent.getParameter('args').projectId;
            
      if (sProjectId) {
        this._showDetail(sProjectId);
      } else {
        this._showMaster();
      }
    },

    _loadProjects: function () {
      const that = this;
            
      jQuery.ajax({
        url: '/api/v2/projects',
        method: 'GET',
        success: function (oData) {
          that.oMasterModel.setProperty('/projects', oData.projects || []);
        },
        error: function (oError) {
          console.error('Failed to load projects:', oError);
          MessageToast.show('Failed to load projects');
        }
      });
    },

    _showMaster: function () {
      this.oLayoutModel.setProperty('/layout', LayoutType.OneColumn);
    },

    _showDetail: function (sProjectId) {
      this.oLayoutModel.setProperty('/layout', LayoutType.TwoColumnsMidExpanded);
      this._loadProjectDetails(sProjectId);
            
      // Select item in master list
      const oList = this.byId('masterList');
      const aItems = oList.getItems();
            
      aItems.forEach((oItem) => {
        const sItemProjectId = oItem.getCustomData()[0].getValue();
        if (sItemProjectId === sProjectId) {
          oList.setSelectedItem(oItem);
        }
      });
    },

    _loadProjectDetails: function (sProjectId) {
      const that = this;
            
      jQuery.ajax({
        url: `/api/v2/projects/${  sProjectId}`,
        method: 'GET',
        success: function (oData) {
          // Add mock data for demonstration
          oData.metrics = {
            success_rate: '98.5%',
            avg_response_time: '245ms',
            total_requests: '12.4K',
            error_rate: '1.5%'
          };
                    
          oData.agents = oData.agents || [
            {
              name: 'Data Processor Agent',
              type: 'Processing',
              status: 'active',
              description: 'Processes incoming data streams',
              last_run: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString()
            },
            {
              name: 'Workflow Orchestrator',
              type: 'Orchestration',
              status: 'active',
              description: 'Manages workflow execution',
              last_run: new Date(Date.now() - 30 * 60 * 1000).toISOString()
            }
          ];
                    
          oData.recent_activity = [
            {
              user: 'System',
              description: 'Project deployed successfully',
              timestamp: new Date(Date.now() - 60 * 60 * 1000).toISOString(),
              type: 'deployment'
            },
            {
              user: 'Developer',
              description: 'Agent configuration updated',
              timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000).toISOString(),
              type: 'configuration'
            }
          ];
                    
          that.oDetailModel.setData(oData);
        },
        error: function (oError) {
          console.error('Failed to load project details:', oError);
          MessageToast.show('Failed to load project details');
        }
      });
    },

    // Master List Events
    onSearch: function (oEvent) {
      const sQuery = oEvent.getParameter('newValue');
      const oList = this.byId('masterList');
      const oBinding = oList.getBinding('items');
            
      if (sQuery && sQuery.length > 0) {
        const aFilters = [
          new Filter('name', FilterOperator.Contains, sQuery),
          new Filter('description', FilterOperator.Contains, sQuery),
          new Filter('type', FilterOperator.Contains, sQuery)
        ];
        const oFilter = new Filter({
          filters: aFilters,
          and: false
        });
        oBinding.filter([oFilter]);
      } else {
        oBinding.filter([]);
      }
    },

    onSelectionChange: function (oEvent) {
      const oSelectedItem = oEvent.getParameter('listItem');
      if (oSelectedItem) {
        const sProjectId = oSelectedItem.getCustomData()[0].getValue();
        this.oRouter.navTo('projectMasterDetail', {
          projectId: sProjectId
        });
      }
    },

    onItemPress: function (oEvent) {
      const oItem = oEvent.getSource();
      const sProjectId = oItem.getCustomData()[0].getValue();
            
      this.oRouter.navTo('projectMasterDetail', {
        projectId: sProjectId
      });
    },

    // Layout Events
    onStateChanged: function (oEvent) {
      const _bIsNavigationArrow = oEvent.getParameter('isNavigationArrow');
      const sLayout = oEvent.getParameter('layout');
            
      this.oLayoutModel.setProperty('/layout', sLayout);
            
      // Update action buttons info
      const oActionButtonsInfo = {
        midColumn: {
          fullScreen: sLayout === LayoutType.MidColumnFullScreen
        }
      };
      this.oLayoutModel.setProperty('/actionButtonsInfo', oActionButtonsInfo);
    },

    onDetailNavButtonPress: function () {
      this.oLayoutModel.setProperty('/layout', LayoutType.OneColumn);
      this.oRouter.navTo('projectMasterDetail');
    },

    onEndNavButtonPress: function () {
      this.oLayoutModel.setProperty('/layout', LayoutType.TwoColumnsMidExpanded);
    },

    // Action Events
    onCreateProject: function () {
      this.oRouter.navTo('projectCreate');
    },

    onRefresh: function () {
      this._loadProjects();
      MessageToast.show('Projects refreshed');
    },

    onEditProject: function () {
      const oProject = this.oDetailModel.getData();
      this.oRouter.navTo('projectEdit', {
        projectId: oProject.project_id
      });
    },

    onDeployProject: function () {
      const oProject = this.oDetailModel.getData();
            
      MessageBox.confirm(
        `Are you sure you want to deploy project '${  oProject.name  }'?`,
        {
          title: 'Deploy Project',
          onClose: function (sAction) {
            if (sAction === MessageBox.Action.OK) {
              this._deployProject(oProject.project_id);
            }
          }.bind(this)
        }
      );
    },

    _deployProject: function (sProjectId) {
      const that = this;
            
      jQuery.ajax({
        url: `/api/v2/projects/${  sProjectId  }/deploy`,
        method: 'POST',
        success: function () {
          MessageToast.show('Project deployment started');
          that._loadProjectDetails(sProjectId);
        },
        error: function (oError) {
          console.error('Failed to deploy project:', oError);
          MessageToast.show('Failed to deploy project');
        }
      });
    },

    onMoreActions: function () {
      MessageToast.show('More actions menu');
    },

    onToggleFullScreen: function () {
      const sCurrentLayout = this.oLayoutModel.getProperty('/layout');
      const sNewLayout = sCurrentLayout === LayoutType.MidColumnFullScreen 
        ? LayoutType.TwoColumnsMidExpanded 
        : LayoutType.MidColumnFullScreen;
            
      this.oLayoutModel.setProperty('/layout', sNewLayout);
    },

    // Agent Actions
    onAddAgent: function () {
      const oProject = this.oDetailModel.getData();
      this.oRouter.navTo('agentBuilder', {
        projectId: oProject.project_id
      });
    },

    onAgentPress: function (oEvent) {
      const oBindingContext = oEvent.getSource().getBindingContext('detailModel');
      const oAgent = oBindingContext.getObject();
            
      // Show agent details in end column
      this.oLayoutModel.setProperty('/layout', LayoutType.ThreeColumnsMidExpanded);
            
      // Load agent details (placeholder)
      MessageToast.show(`Loading agent: ${  oAgent.name}`);
    },

    onRunAgent: function (oEvent) {
      const oBindingContext = oEvent.getSource().getBindingContext('detailModel');
      const oAgent = oBindingContext.getObject();
            
      MessageToast.show(`Running agent: ${  oAgent.name}`);
    },

    onEditAgent: function (oEvent) {
      const oBindingContext = oEvent.getSource().getBindingContext('detailModel');
      const oAgent = oBindingContext.getObject();
            
      MessageToast.show(`Editing agent: ${  oAgent.name}`);
    },

    onViewAllActivity: function () {
      MessageToast.show('View all activity');
    },

    // Formatters
    formatDate: function (sDate) {
      if (!sDate) {
        return '';
      }
            
      const oDate = new Date(sDate);
      return oDate.toLocaleDateString();
    },

    formatRelativeTime: function (sDate) {
      if (!sDate) {
        return '';
      }
            
      const oDate = new Date(sDate);
      const oNow = new Date();
      const iDiff = oNow.getTime() - oDate.getTime();
      const iMinutes = Math.floor(iDiff / (1000 * 60));
      const iHours = Math.floor(iMinutes / 60);
      const iDays = Math.floor(iHours / 24);
            
      if (iMinutes < 1) {
        return 'Just now';
      } else if (iMinutes < 60) {
        return `${iMinutes  } minutes ago`;
      } else if (iHours < 24) {
        return `${iHours  } hours ago`;
      } else if (iDays === 1) {
        return 'Yesterday';
      } else if (iDays < 7) {
        return `${iDays  } days ago`;
      } else {
        return oDate.toLocaleDateString();
      }
    },

    formatStatusState: function (sStatus) {
      switch (sStatus) {
      case 'active':
        return 'Success';
      case 'deployed':
        return 'Success';
      case 'inactive':
        return 'Warning';
      case 'error':
        return 'Error';
      default:
        return 'None';
      }
    },

    formatStatusIcon: function (sStatus) {
      switch (sStatus) {
      case 'active':
        return 'sap-icon://status-positive';
      case 'deployed':
        return 'sap-icon://status-positive';
      case 'inactive':
        return 'sap-icon://status-inactive';
      case 'error':
        return 'sap-icon://status-negative';
      default:
        return '';
      }
    },

    formatAgentsState: function (iCount) {
      if (iCount > 10) {
        return 'Success';
      } else if (iCount > 5) {
        return 'Warning';
      } else {
        return 'None';
      }
    },

    formatDeploymentState: function (sStatus) {
      switch (sStatus) {
      case 'deployed':
        return 'Success';
      case 'deploying':
        return 'Warning';
      case 'failed':
        return 'Error';
      default:
        return 'None';
      }
    },

    formatAgentStatusState: function (sStatus) {
      switch (sStatus) {
      case 'active':
        return 'Success';
      case 'inactive':
        return 'Warning';
      case 'error':
        return 'Error';
      default:
        return 'None';
      }
    },

    formatActivityIcon: function (sType) {
      switch (sType) {
      case 'deployment':
        return 'sap-icon://cloud';
      case 'configuration':
        return 'sap-icon://settings';
      case 'execution':
        return 'sap-icon://play';
      default:
        return 'sap-icon://information';
      }
    }
  });
});
