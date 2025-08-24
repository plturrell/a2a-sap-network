sap.ui.define([
  './BaseController',
  'sap/ui/model/json/JSONModel',
  'sap/m/MessageToast',
  'sap/m/MessageBox',
  'sap/ui/core/routing/History'
], (BaseController, JSONModel, MessageToast, MessageBox, History) => {
  'use strict';

  return BaseController.extend('a2a.portal.controller.ProjectObjectPage', {

    onInit: function () {
      // Call parent init
      // eslint-disable-next-line prefer-rest-params
      BaseController.prototype.onInit.apply(this, arguments);
            
      // Initialize view model
      const oViewModel = new JSONModel({
        busy: false,
        delay: 0,
        editMode: false
      });
      this.setModel(oViewModel, 'view');

      // Initialize project model
      this.setModel(new JSONModel(), 'project');

      // Get router and attach to routes
      const oRouter = this.getOwnerComponent().getRouter();
      oRouter.getRoute('projectObjectPage').attachPatternMatched(this._onObjectMatched, this);
    },

    _onObjectMatched: function (oEvent) {
      const sProjectId = oEvent.getParameter('args').projectId;
      this._projectId = sProjectId;
      this._loadProjectDetails(sProjectId);
    },

    _loadProjectDetails: function (sProjectId) {
      const oModel = this.getModel('project');
      this.getModel('view').setProperty('/busy', true);

      // In a real app, this would be an API call
      jQuery.ajax({
        url: `/api/projects/${  sProjectId  }?expand=agents,workflows,members,metrics,activities`,
        method: 'GET',
        success: function (data) {
          // Transform data for UI
          data = this._enrichProjectData(data);
          oModel.setData(data);
          this.getModel('view').setProperty('/busy', false);
        }.bind(this),
        error: function () {
          // Use mock data as fallback
          const oMockData = this._getMockProjectData(sProjectId);
          oModel.setData(oMockData);
          this.getModel('view').setProperty('/busy', false);
        }.bind(this)
      });
    },

    _enrichProjectData: function (oData) {
      // Add calculated fields
      oData.progress = this._calculateProgress(oData);
      oData.budgetStatus = oData.budget > oData.budgetLimit ? 'OVER' : 'OK';
            
      // Enrich activities with icons
      if (oData.activities) {
        oData.activities.forEach((activity) => {
          activity.icon = this._getActivityIcon(activity.type);
        });
      }

      return oData;
    },

    _calculateProgress: function (oProject) {
      if (!oProject.startDate || !oProject.endDate) {
        return 0;
      }
            
      const start = new Date(oProject.startDate).getTime();
      const end = new Date(oProject.endDate).getTime();
      const now = Date.now();
            
      if (now < start) {
        return 0;
      }
      if (now > end) {
        return 100;
      }
            
      return Math.round(((now - start) / (end - start)) * 100);
    },

    _getActivityIcon: function (sType) {
      const mIcons = {
        'deployment': 'sap-icon://upload',
        'agent_created': 'sap-icon://add',
        'workflow_executed': 'sap-icon://process',
        'member_added': 'sap-icon://employee',
        'status_changed': 'sap-icon://status-positive',
        'error': 'sap-icon://error'
      };
      return mIcons[sType] || 'sap-icon://activity-items';
    },

    _getMockProjectData: function (sProjectId) {
      return {
        projectId: sProjectId,
        name: 'Customer Analytics Platform',
        description: 'Enterprise-scale multi-agent system for real-time customer behavior analysis and predictive insights',
        status: 'ACTIVE',
        priority: 'HIGH',
        progress: 65,
        budget: 250000,
        budgetLimit: 300000,
        currency: 'EUR',
        budgetStatus: 'OK',
        startDate: new Date('2024-01-01'),
        endDate: new Date('2024-12-31'),
        costCenter: 'CC-IT-001',
        businessUnit: {
          id: 'BU_001',
          name: 'Digital Innovation'
        },
        department: {
          id: 'DEPT_IT_001',
          name: 'Information Technology'
        },
        projectManager: {
          id: 'USR_001',
          displayName: 'Sarah Johnson',
          email: 'sarah.johnson@company.com'
        },
        agents: [
          {
            agentId: 'agent_001',
            name: 'Data Ingestion Agent',
            type: 'reactive',
            status: 'DEPLOYED',
            healthStatus: 'HEALTHY',
            executionCount: 1543
          },
          {
            agentId: 'agent_002',
            name: 'Analytics Processing Agent',
            type: 'proactive',
            status: 'DEPLOYED',
            healthStatus: 'HEALTHY',
            executionCount: 892
          },
          {
            agentId: 'agent_003',
            name: 'Notification Agent',
            type: 'reactive',
            status: 'TESTING',
            healthStatus: 'UNKNOWN',
            executionCount: 0
          }
        ],
        workflows: [
          {
            id: 'wf_001',
            name: 'Customer Journey Analysis',
            description: 'End-to-end customer journey tracking and analysis',
            version: '1.2.0',
            status: 'PUBLISHED'
          },
          {
            id: 'wf_002',
            name: 'Predictive Churn Analysis',
            description: 'ML-based customer churn prediction workflow',
            version: '1.0.0',
            status: 'DRAFT'
          }
        ],
        members: [
          {
            user: {
              id: 'USR_001',
              displayName: 'Sarah Johnson'
            },
            role: 'OWNER',
            joinedDate: new Date('2024-01-01')
          },
          {
            user: {
              id: 'USR_002',
              displayName: 'Michael Chen'
            },
            role: 'DEVELOPER',
            joinedDate: new Date('2024-01-15')
          },
          {
            user: {
              id: 'USR_003',
              displayName: 'Emma Davis'
            },
            role: 'TESTER',
            joinedDate: new Date('2024-02-01')
          }
        ],
        metrics: {
          successRate: 96.5,
          avgResponseTime: 342,
          executionCount: 2435,
          errorRate: 3.5,
          uptime: 99.9
        },
        activities: [
          {
            timestamp: new Date(),
            title: 'Agent Deployed',
            description: 'Analytics Processing Agent deployed to production',
            user: 'Michael Chen',
            type: 'deployment'
          },
          {
            timestamp: new Date(Date.now() - 3600000),
            title: 'Workflow Executed',
            description: 'Customer Journey Analysis workflow completed successfully',
            user: 'System',
            type: 'workflow_executed'
          },
          {
            timestamp: new Date(Date.now() - 7200000),
            title: 'Member Added',
            description: 'Emma Davis joined as Tester',
            user: 'Sarah Johnson',
            type: 'member_added'
          }
        ]
      };
    },

    onEdit: function () {
      const bEditMode = this.getModel('view').getProperty('/editMode');
      this.getModel('view').setProperty('/editMode', !bEditMode);
            
      if (!bEditMode) {
        MessageToast.show('Edit mode enabled');
      } else {
        this._saveChanges();
      }
    },

    _saveChanges: function () {
      // In a real app, this would save to backend
      MessageToast.show('Changes saved successfully');
    },

    onDeploy: function () {
      MessageBox.confirm(
        'Deploy this project to production?', {
          title: 'Confirm Deployment',
          onClose: function (sAction) {
            if (sAction === MessageBox.Action.OK) {
              this._deployProject();
            }
          }.bind(this)
        }
      );
    },

    _deployProject: function () {
      // Simulate deployment
      MessageToast.show('Deployment initiated...');
            
      setTimeout(() => {
        this.getModel('project').setProperty('/status', 'DEPLOYED');
        MessageToast.show('Project deployed successfully!');
      }, 2000);
    },

    onArchive: function () {
      MessageBox.warning(
        'Archive this project? It will no longer be active.', {
          title: 'Archive Project',
          actions: [MessageBox.Action.YES, MessageBox.Action.NO],
          onClose: function (sAction) {
            if (sAction === MessageBox.Action.YES) {
              this._archiveProject();
            }
          }.bind(this)
        }
      );
    },

    _archiveProject: function () {
      this.getModel('project').setProperty('/status', 'ARCHIVED');
      MessageToast.show('Project archived');
    },

    onActions: function (oEvent) {
      const oButton = oEvent.getSource();
            
      if (!this._actionSheet) {
        this._actionSheet = sap.ui.xmlfragment(
          'a2a.portal.view.fragments.ProjectActions',
          this
        );
        this.getView().addDependent(this._actionSheet);
      }
            
      this._actionSheet.openBy(oButton);
    },

    onAddAgent: function () {
      this.getRouter().navTo('agentBuilder', {
        projectId: this._projectId
      });
    },

    onAgentPress: function (oEvent) {
      const oAgent = oEvent.getSource().getBindingContext('project').getObject();
      MessageToast.show(`Agent details: ${  oAgent.name}`);
    },

    onDesignWorkflow: function () {
      this.getRouter().navTo('bpmnDesigner', {
        projectId: this._projectId
      });
    },

    onWorkflowPress: function (oEvent) {
      const oWorkflow = oEvent.getSource().getBindingContext('project').getObject();
      MessageToast.show(`Workflow details: ${  oWorkflow.name}`);
    },

    onAddMember: function () {
      MessageToast.show('Add team member dialog');
    },

    onMemberPress: function (oEvent) {
      const oMember = oEvent.getSource().getBindingContext('project').getObject();
      MessageToast.show(`Member details: ${  oMember.user.displayName}`);
    },

    onManagerPress: function () {
      const oManager = this.getModel('project').getProperty('/projectManager');
      MessageToast.show(`Manager: ${  oManager.email}`);
    },

    formatAgentStatus: function (sStatus) {
      const mStates = {
        'DEPLOYED': 'Success',
        'TESTING': 'Warning',
        'FAILED': 'Error',
        'DRAFT': 'None'
      };
      return mStates[sStatus] || 'None';
    },

    formatHealthStatus: function (sHealth) {
      const mStates = {
        'HEALTHY': 'Success',
        'DEGRADED': 'Warning',
        'UNHEALTHY': 'Error',
        'UNKNOWN': 'None'
      };
      return mStates[sHealth] || 'None';
    },

    formatHealthIcon: function (sHealth) {
      const mIcons = {
        'HEALTHY': 'sap-icon://status-positive',
        'DEGRADED': 'sap-icon://status-critical',
        'UNHEALTHY': 'sap-icon://status-negative',
        'UNKNOWN': 'sap-icon://question-mark'
      };
      return mIcons[sHealth] || 'sap-icon://question-mark';
    },

    onNavBack: function () {
      const oHistory = History.getInstance();
      const sPreviousHash = oHistory.getPreviousHash();

      if (sPreviousHash !== undefined) {
        window.history.go(-1);
      } else {
        this.getRouter().navTo('projects', {}, true);
      }
    }
  });
});