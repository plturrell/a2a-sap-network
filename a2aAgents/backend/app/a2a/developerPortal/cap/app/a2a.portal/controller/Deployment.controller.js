sap.ui.define([
  'sap/ui/core/mvc/Controller',
  'sap/ui/model/json/JSONModel',
  'sap/m/MessageToast',
  'sap/m/MessageBox',
  'sap/ui/core/format/DateFormat'
], (Controller, JSONModel, MessageToast, MessageBox, DateFormat) => {
  'use strict';

  return Controller.extend('a2a.portal.controller.Deployment', {

    onInit: function () {
      // Initialize view model
      const oViewModel = new JSONModel({
        viewMode: 'overview',
        environments: [],
        pipelines: [],
        releases: [],
        stats: {},
        busy: false
      });
      this.getView().setModel(oViewModel, 'view');

      // Load deployment data
      this._loadDeploymentData();
    },

    _loadDeploymentData: function () {
      const oViewModel = this.getView().getModel('view');
      oViewModel.setProperty('/busy', true);

      jQuery.ajax({
        url: '/api/deployment/data',
        method: 'GET',
        success: function (data) {
          oViewModel.setProperty('/environments', data.environments || []);
          oViewModel.setProperty('/pipelines', data.pipelines || []);
          oViewModel.setProperty('/releases', data.releases || []);
          oViewModel.setProperty('/stats', data.stats || {});
          oViewModel.setProperty('/busy', false);
        }.bind(this),
        error: function (_xhr, _status, _error) {
          // Fallback to mock data
          const oMockData = this._getMockDeploymentData();
          oViewModel.setProperty('/environments', oMockData.environments);
          oViewModel.setProperty('/pipelines', oMockData.pipelines);
          oViewModel.setProperty('/releases', oMockData.releases);
          oViewModel.setProperty('/stats', oMockData.stats);
          oViewModel.setProperty('/busy', false);
          MessageToast.show('Using sample data - backend connection unavailable');
        }.bind(this)
      });
    },

    _getMockDeploymentData: function () {
      return {
        stats: {
          activeDeployments: 12,
          successRate: 96.5,
          pendingReleases: 3,
          avgDeployTime: 8.5
        },
        environments: [
          {
            id: 'env-1',
            name: 'Production',
            description: 'Live production environment',
            type: 'production',
            activeAgents: 8,
            lastDeployment: '2024-01-22T14:30:00Z',
            status: 'healthy'
          },
          {
            id: 'env-2',
            name: 'Staging',
            description: 'Pre-production staging environment',
            type: 'staging',
            activeAgents: 6,
            lastDeployment: '2024-01-22T12:15:00Z',
            status: 'healthy'
          },
          {
            id: 'env-3',
            name: 'Development',
            description: 'Development and testing environment',
            type: 'development',
            activeAgents: 4,
            lastDeployment: '2024-01-22T10:45:00Z',
            status: 'warning'
          }
        ],
        pipelines: [
          {
            id: 'pipeline-1',
            name: 'Agent0 CI/CD Pipeline',
            description: 'Automated build and deployment for Agent0',
            trigger: 'Git Push',
            lastRun: '2024-01-22T14:00:00Z',
            duration: 12.5,
            status: 'success'
          },
          {
            id: 'pipeline-2',
            name: 'Agent1 CI/CD Pipeline',
            description: 'Automated build and deployment for Agent1',
            trigger: 'Scheduled',
            lastRun: '2024-01-22T11:00:00Z',
            duration: 8.2,
            status: 'success'
          },
          {
            id: 'pipeline-3',
            name: 'Integration Pipeline',
            description: 'End-to-end integration testing',
            trigger: 'Manual',
            lastRun: '2024-01-21T16:30:00Z',
            duration: 25.8,
            status: 'failed'
          }
        ],
        releases: [
          {
            id: 'release-1',
            version: 'v2.1.0',
            codename: 'Winter Release',
            agentName: 'Agent0 Data Product',
            changeCount: 15,
            createdBy: 'Release Manager',
            creationDate: '2024-01-20T09:00:00Z',
            status: 'approved'
          },
          {
            id: 'release-2',
            version: 'v1.8.3',
            codename: 'Patch Release',
            agentName: 'Agent1 Standardization',
            changeCount: 3,
            createdBy: 'Dev Team',
            creationDate: '2024-01-19T14:20:00Z',
            status: 'deployed'
          },
          {
            id: 'release-3',
            version: 'v1.2.0-beta',
            codename: 'Beta Release',
            agentName: 'Integration Agent',
            changeCount: 8,
            createdBy: 'QA Team',
            creationDate: '2024-01-18T11:15:00Z',
            status: 'draft'
          }
        ]
      };
    },

    onRefresh: function () {
      this._loadDeploymentData();
      MessageToast.show('Deployment data refreshed');
    },

    onViewChange: function (oEvent) {
      const sSelectedKey = oEvent.getParameter('item').getKey();
      const oViewModel = this.getView().getModel('view');
      oViewModel.setProperty('/viewMode', sSelectedKey);
    },

    onSearch: function (oEvent) {
      const sQuery = oEvent.getParameter('query');
      const sViewMode = this.getView().getModel('view').getProperty('/viewMode');
      let oTable, oBinding;

      switch (sViewMode) {
      case 'environments':
        oTable = this.byId('environmentsTable');
        break;
      case 'pipelines':
        oTable = this.byId('pipelinesTable');
        break;
      case 'releases':
        oTable = this.byId('releasesTable');
        break;
      default:
        return;
      }

      if (oTable) {
        oBinding = oTable.getBinding('items');
        if (sQuery && sQuery.length > 0) {
          const oFilter = new sap.ui.model.Filter([
            new sap.ui.model.Filter('name', sap.ui.model.FilterOperator.Contains, sQuery),
            new sap.ui.model.Filter('description', sap.ui.model.FilterOperator.Contains, sQuery)
          ], false);
          oBinding.filter([oFilter]);
        } else {
          oBinding.filter([]);
        }
      }
    },

    onOpenFilterDialog: function () {
      MessageToast.show('Filter dialog - coming soon');
    },

    onOpenSortDialog: function () {
      MessageToast.show('Sort dialog - coming soon');
    },

    onDeployAgent: function () {
      if (!this._oDeployDialog) {
        this._oDeployDialog = sap.ui.xmlfragment('a2a.portal.fragment.DeployAgentDialog', this);
        this.getView().addDependent(this._oDeployDialog);
      }
      this._oDeployDialog.open();
    },

    onCreatePipeline: function () {
      if (!this._oCreatePipelineDialog) {
        this._oCreatePipelineDialog = sap.ui.xmlfragment('a2a.portal.fragment.CreatePipelineDialog', this);
        this.getView().addDependent(this._oCreatePipelineDialog);
      }
      this._oCreatePipelineDialog.open();
    },

    onDeployToStaging: function () {
      MessageToast.show('Deploying to staging environment...');
    },

    onPromoteToProduction: function () {
      MessageBox.confirm(
        'Promote staging deployment to production?', {
          title: 'Promote to Production',
          onClose: function (sAction) {
            if (sAction === MessageBox.Action.OK) {
              MessageToast.show('Promoting to production...');
            }
          }
        }
      );
    },

    onRollbackRelease: function () {
      MessageBox.confirm(
        'Rollback to previous release? This will revert recent changes.', {
          icon: MessageBox.Icon.WARNING,
          title: 'Rollback Release',
          onClose: function (sAction) {
            if (sAction === MessageBox.Action.OK) {
              MessageToast.show('Rolling back release...');
            }
          }
        }
      );
    },

    onEnvironmentPress: function (oEvent) {
      const oContext = oEvent.getSource().getBindingContext('view');
      const sEnvironmentId = oContext.getProperty('id');
      MessageToast.show(`Environment selected: ${  sEnvironmentId}`);
    },

    onCreateEnvironment: function () {
      if (!this._oCreateEnvDialog) {
        this._oCreateEnvDialog = sap.ui.xmlfragment('a2a.portal.fragment.CreateEnvironmentDialog', this);
        this.getView().addDependent(this._oCreateEnvDialog);
      }
      this._oCreateEnvDialog.open();
    },

    onManageEnvironment: function (oEvent) {
      oEvent.stopPropagation();
      const oContext = oEvent.getSource().getBindingContext('view');
      const sEnvName = oContext.getProperty('name');
      MessageToast.show(`Managing environment: ${  sEnvName}`);
    },

    onViewLogs: function (oEvent) {
      oEvent.stopPropagation();
      const oContext = oEvent.getSource().getBindingContext('view');
      const sEnvName = oContext.getProperty('name');
      MessageToast.show(`Opening logs for: ${  sEnvName}`);
    },

    onDeleteEnvironment: function (oEvent) {
      oEvent.stopPropagation();
      const oContext = oEvent.getSource().getBindingContext('view');
      const sEnvName = oContext.getProperty('name');
      const sEnvId = oContext.getProperty('id');
            
      MessageBox.confirm(
        `Delete environment '${  sEnvName  }'? This action cannot be undone.`, {
          icon: MessageBox.Icon.WARNING,
          title: 'Confirm Deletion',
          onClose: function (sAction) {
            if (sAction === MessageBox.Action.OK) {
              this._deleteEnvironment(sEnvId);
            }
          }.bind(this)
        }
      );
    },

    _deleteEnvironment: function (sEnvId) {
      jQuery.ajax({
        url: `/api/deployment/environments/${  sEnvId}`,
        method: 'DELETE',
        success: function () {
          MessageToast.show('Environment deleted successfully');
          this._loadDeploymentData();
        }.bind(this),
        error: function (xhr, status, error) {
          MessageToast.show(`Failed to delete environment: ${  error}`);
        }.bind(this)
      });
    },

    onPipelinePress: function (oEvent) {
      const oContext = oEvent.getSource().getBindingContext('view');
      const sPipelineId = oContext.getProperty('id');
      MessageToast.show(`Pipeline selected: ${  sPipelineId}`);
    },

    onRunPipeline: function (oEvent) {
      oEvent.stopPropagation();
      const oContext = oEvent.getSource().getBindingContext('view');
      const sPipelineName = oContext.getProperty('name');
      MessageToast.show(`Running pipeline: ${  sPipelineName}`);
    },

    onEditPipeline: function (oEvent) {
      oEvent.stopPropagation();
      const oContext = oEvent.getSource().getBindingContext('view');
      const sPipelineName = oContext.getProperty('name');
      MessageToast.show(`Editing pipeline: ${  sPipelineName}`);
    },

    onViewHistory: function (oEvent) {
      oEvent.stopPropagation();
      const oContext = oEvent.getSource().getBindingContext('view');
      const sPipelineName = oContext.getProperty('name');
      MessageToast.show(`Viewing history for: ${  sPipelineName}`);
    },

    onReleasePress: function (oEvent) {
      const oContext = oEvent.getSource().getBindingContext('view');
      const sReleaseId = oContext.getProperty('id');
      MessageToast.show(`Release selected: ${  sReleaseId}`);
    },

    onCreateRelease: function () {
      if (!this._oCreateReleaseDialog) {
        this._oCreateReleaseDialog = sap.ui.xmlfragment('a2a.portal.fragment.CreateReleaseDialog', this);
        this.getView().addDependent(this._oCreateReleaseDialog);
      }
      this._oCreateReleaseDialog.open();
    },

    onDeployRelease: function (oEvent) {
      oEvent.stopPropagation();
      const oContext = oEvent.getSource().getBindingContext('view');
      const sVersion = oContext.getProperty('version');
            
      MessageBox.confirm(
        `Deploy release ${  sVersion  } to production?`, {
          title: 'Deploy Release',
          onClose: function (sAction) {
            if (sAction === MessageBox.Action.OK) {
              MessageToast.show(`Deploying release ${  sVersion  }...`);
            }
          }
        }
      );
    },

    onViewReleaseDetails: function (oEvent) {
      oEvent.stopPropagation();
      const oContext = oEvent.getSource().getBindingContext('view');
      const sVersion = oContext.getProperty('version');
      MessageToast.show(`Viewing details for release: ${  sVersion}`);
    },

    onRunSelected: function () {
      MessageToast.show('Running selected pipelines...');
    },

    onExportSelected: function () {
      MessageToast.show('Export functionality - coming soon');
    },

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
      case 'healthy': case 'success': case 'approved': case 'deployed': return 'Success';
      case 'warning': case 'draft': return 'Warning';
      case 'error': case 'failed': return 'Error';
      default: return 'None';
      }
    }
  });
});