sap.ui.define([
  './BaseController',
  'sap/ui/model/json/JSONModel',
  'sap/m/MessageToast',
  'sap/m/MessageBox',
  'sap/ui/core/routing/History'
], (BaseController, JSONModel, MessageToast, MessageBox, History) => {
  'use strict';
  /* global localStorage, Notification, sap, jQuery, Blob, $, URL */

  return BaseController.extend('a2a.portal.controller.ProjectObjectPage', {

    onInit: function () {
      // Call parent init
      BaseController.prototype.onInit.apply(this, arguments);
            
      // Initialize view model
      const oViewModel = new JSONModel({
        busy: false,
        delay: 0,
        editMode: false,
        currentSection: null,
        anchorBarVisible: true,
        anchorBarItems: [],
        fieldEditability: {
          enabled: true,
          userRole: 'PROJECT_MANAGER', // Can be: ADMIN, PROJECT_MANAGER, DEVELOPER, VIEWER
          projectStatus: 'ACTIVE',
          editableFields: {},
          readOnlyFields: {},
          conditionalFields: {},
          rolePermissions: {
            'ADMIN': ['name', 'description', 'startDate', 'endDate', 'budget', 'costCenter', 'priority', 'status'],
            'PROJECT_MANAGER': ['name', 'description', 'startDate', 'endDate', 'budget', 'costCenter'],
            'DEVELOPER': ['description'],
            'VIEWER': []
          },
          statusRestrictions: {
            'ARCHIVED': [],
            'DEPLOYED': ['name', 'startDate', 'endDate'],
            'ACTIVE': 'ALL',
            'DRAFT': 'ALL'
          }
        },
        insights: {
          enabled: true,
          autoRefresh: true,
          refreshInterval: 30000, // 30 seconds
          showInSidebar: false,
          showInModal: true,
          currentInsights: [],
          insightCategories: {
            risk: { enabled: true, priority: 1 },
            optimization: { enabled: true, priority: 2 },
            compliance: { enabled: true, priority: 3 },
            prediction: { enabled: true, priority: 4 },
            recommendation: { enabled: true, priority: 5 }
          },
          analysisConfig: {
            enableRealTimeAnalysis: true,
            includeHistoricalData: true,
            enableComparativeAnalysis: true,
            enablePredictiveAnalysis: true,
            confidenceThreshold: 0.7
          },
          displayConfig: {
            maxInsightsPerCategory: 3,
            showConfidenceScores: true,
            enableDetailedView: true,
            showActionableRecommendations: true,
            groupByPriority: true
          }
        },
        suggestions: {
          enabled: true,
          autoShow: true,
          showDelay: 1000,
          maxSuggestions: 5,
          contextAware: true,
          learningEnabled: true,
          currentSuggestions: [],
          activeSuggestion: null,
          userPreferences: {
            enableAI: true,
            showValidationSuggestions: true,
            showCompletionSuggestions: true,
            showOptimizationSuggestions: true
          },
          feedback: {
            accepted: [],
            rejected: [],
            applied: []
          }
        },
        sectionHighlighting: {
          enabled: true,
          animationDuration: 200,
          highlightClass: 'a2a-section-highlighted',
          fadeClass: 'a2a-section-fade',
          currentHighlighted: null
        },
        sectionCollapse: {
          enabled: true,
          animationDuration: 300,
          collapsedSections: [],
          persistState: true,
          allowMultipleExpanded: true
        },
        lazyLoading: {
          enabled: true,
          threshold: 0.1,
          loadedSections: [],
          preloadDistance: 200,
          enablePlaceholders: true,
          loadingTimeout: 5000
        }
      });
      this.setModel(oViewModel, 'view');

      // Initialize project model
      this.setModel(new JSONModel(), 'project');

      // Get router and attach to routes
      const oRouter = this.getOwnerComponent().getRouter();
      oRouter.getRoute('projectObjectPage').attachPatternMatched(this._onObjectMatched, this);
            
      // Initialize anchor bar functionality
      this._initializeAnchorBar();
            
      // Initialize section highlighting
      this._initializeSectionHighlighting();
            
      // Initialize section expand/collapse
      this._initializeSectionCollapse();
            
      // Initialize lazy loading
      this._initializeLazyLoading();
            
      // Initialize field editability
      this._initializeFieldEditability();
            
      // Initialize suggestions system
      this._initializeSuggestions();
            
      // Initialize insights system
      this._initializeInsights();
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
                    
          // Update anchor bar badges with actual data
          this._updateAnchorBarBadges();
        }.bind(this),
        error: function () {
          // Use mock data as fallback
          const oMockData = this._getMockProjectData(sProjectId);
          oModel.setData(oMockData);
          this.getModel('view').setProperty('/busy', false);
                    
          // Update anchor bar badges with mock data
          this._updateAnchorBarBadges();
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
        this._enableEditMode();
      } else {
        this._disableEditMode();
      }
    },
        
    _enableEditMode: function () {
      // Update UI state for edit mode
      this.getModel('view').setProperty('/editMode', true);
      this.getModel('view').setProperty('/editButtonText', 'Save Changes');
      this.getModel('view').setProperty('/editButtonType', 'Emphasized');
      this.getModel('view').setProperty('/editButtonIcon', 'sap-icon://save');
            
      // Start edit mode time tracking
      this._editModeStartTime = Date.now();
            
      // Enable form controls
      this._toggleFormControls(true);
            
      // Show edit mode indicators
      this._showEditModeIndicators(true);
            
      // Store original data for cancel functionality
      this._originalData = jQuery.extend(true, {}, this.getModel('project').getData());
            
      MessageToast.show('Edit mode enabled. You can now modify project details.');
    },
        
    _disableEditMode: function () {
      // Validate changes before saving
      if (this._validateEditForm()) {
        this._saveChanges();
      } else {
        // Revert to edit mode if validation fails
        this.getModel('view').setProperty('/editMode', true);
        return;
      }
            
      // Update UI state back to view mode
      this.getModel('view').setProperty('/editMode', false);
      this.getModel('view').setProperty('/editButtonText', 'Edit');
      this.getModel('view').setProperty('/editButtonType', 'Default');
      this.getModel('view').setProperty('/editButtonIcon', 'sap-icon://edit');
            
      // Disable form controls
      this._toggleFormControls(false);
            
      // Hide edit mode indicators
      this._showEditModeIndicators(false);
            
      // Clear original data backup
      delete this._originalData;
    },
        
    _validateEditForm: function () {
      const oProjectData = this.getModel('project').getData();
      const oValidationResult = this._performComprehensiveValidation(oProjectData);
            
      // Update validation display
      this._updateValidationDisplay(oValidationResult);
            
      // Show validation summary if there are errors
      if (!oValidationResult.isValid) {
        this._showValidationSummary(oValidationResult);
      }
            
      return oValidationResult.isValid;
    },
        
    _performComprehensiveValidation: function (oProjectData) {
      const oResult = {
        isValid: true,
        errors: [],
        warnings: [],
        fieldValidation: {},
        summary: {
          totalErrors: 0,
          totalWarnings: 0,
          validFields: 0,
          totalFields: 0
        }
      };
            
      const aFieldsToValidate = ['name', 'description', 'startDate', 'endDate', 'budget', 'costCenter'];
      oResult.summary.totalFields = aFieldsToValidate.length;
            
      // Validate each field individually
      aFieldsToValidate.forEach((sField) => {
        const oFieldResult = this._validateField(sField, oProjectData[sField], oProjectData);
        oResult.fieldValidation[sField] = oFieldResult;
                
        if (oFieldResult.errors.length > 0) {
          oResult.errors = oResult.errors.concat(oFieldResult.errors);
          oResult.isValid = false;
          oResult.summary.totalErrors += oFieldResult.errors.length;
        } else {
          oResult.summary.validFields++;
        }
                
        if (oFieldResult.warnings.length > 0) {
          oResult.warnings = oResult.warnings.concat(oFieldResult.warnings);
          oResult.summary.totalWarnings += oFieldResult.warnings.length;
        }
      });
            
      // Cross-field validation
      const oCrossFieldResult = this._validateCrossFieldRules(oProjectData);
      if (oCrossFieldResult.errors.length > 0) {
        oResult.errors = oResult.errors.concat(oCrossFieldResult.errors);
        oResult.isValid = false;
        oResult.summary.totalErrors += oCrossFieldResult.errors.length;
      }
      if (oCrossFieldResult.warnings.length > 0) {
        oResult.warnings = oResult.warnings.concat(oCrossFieldResult.warnings);
        oResult.summary.totalWarnings += oCrossFieldResult.warnings.length;
      }
            
      return oResult;
    },
        
    _validateField: function (sFieldName, vValue, oProjectData) {
      let oResult = {
        field: sFieldName,
        value: vValue,
        errors: [],
        warnings: [],
        isValid: true,
        severity: 'None'
      };
            
      switch (sFieldName) {
      case 'name':
        oResult = this._validateProjectName(vValue);
        break;
      case 'description':
        oResult = this._validateDescription(vValue);
        break;
      case 'startDate':
        oResult = this._validateStartDate(vValue, oProjectData);
        break;
      case 'endDate':
        oResult = this._validateEndDate(vValue, oProjectData);
        break;
      case 'budget':
        oResult = this._validateBudget(vValue, oProjectData);
        break;
      case 'costCenter':
        oResult = this._validateCostCenter(vValue);
        break;
      }
            
      oResult.field = sFieldName;
      oResult.isValid = oResult.errors.length === 0;
            
      if (oResult.errors.length > 0) {
        oResult.severity = 'Error';
      } else if (oResult.warnings.length > 0) {
        oResult.severity = 'Warning';
      } else {
        oResult.severity = 'Success';
      }
            
      return oResult;
    },
        
    _validateProjectName: function (sName) {
      const oResult = { errors: [], warnings: [] };
            
      if (!sName || sName.trim() === '') {
        oResult.errors.push('Project name is required');
      } else {
        if (sName.length < 3) {
          oResult.errors.push('Project name must be at least 3 characters long');
        } else if (sName.length > 100) {
          oResult.errors.push('Project name cannot exceed 100 characters');
        }
                
        if (sName.length < 5) {
          oResult.warnings.push('Project name is quite short - consider a more descriptive name');
        }
                
        // Check for special characters
        if (!/^[a-zA-Z0-9\s\-_\.]+$/.test(sName)) {
          oResult.errors.push('Project name contains invalid characters');
        }
                
        // Check for reserved words
        const aReservedWords = ['admin', 'system', 'test', 'demo'];
        if (aReservedWords.some(word => sName.toLowerCase().includes(word))) {
          oResult.warnings.push('Project name contains reserved words');
        }
      }
            
      return oResult;
    },
        
    _validateDescription: function (sDescription) {
      const oResult = { errors: [], warnings: [] };
            
      if (!sDescription || sDescription.trim() === '') {
        oResult.errors.push('Project description is required');
      } else {
        if (sDescription.length < 10) {
          oResult.warnings.push('Description is quite short - provide more details');
        } else if (sDescription.length > 2000) {
          oResult.errors.push('Description cannot exceed 2000 characters');
        }
                
        // Check for meaningful content
        if (sDescription.trim().split(/\s+/).length < 5) {
          oResult.warnings.push('Description should contain more detailed information');
        }
      }
            
      return oResult;
    },
        
    _validateStartDate: function (dStartDate, oProjectData) {
      const oResult = { errors: [], warnings: [] };
            
      if (dStartDate) {
        const startDate = new Date(dStartDate);
        const now = new Date();
        const yesterday = new Date(now);
        yesterday.setDate(yesterday.getDate() - 1);
                
        if (isNaN(startDate.getTime())) {
          oResult.errors.push('Invalid start date format');
        } else {
          if (startDate < yesterday) {
            oResult.warnings.push('Start date is in the past');
          }
                    
          const futureLimit = new Date(now);
          futureLimit.setFullYear(futureLimit.getFullYear() + 5);
          if (startDate > futureLimit) {
            oResult.warnings.push('Start date is more than 5 years in the future');
          }
        }
      }
            
      return oResult;
    },
        
    _validateEndDate: function (dEndDate, oProjectData) {
      const oResult = { errors: [], warnings: [] };
            
      if (dEndDate) {
        const endDate = new Date(dEndDate);
                
        if (isNaN(endDate.getTime())) {
          oResult.errors.push('Invalid end date format');
        } else if (oProjectData.startDate) {
          const startDate = new Date(oProjectData.startDate);
                    
          if (endDate <= startDate) {
            oResult.errors.push('End date must be after start date');
          } else {
            const daysDiff = (endDate - startDate) / (1000 * 60 * 60 * 24);
            if (daysDiff < 7) {
              oResult.errors.push('Project duration must be at least 7 days');
            } else if (daysDiff > 1095) { // 3 years
              oResult.warnings.push('Project duration exceeds 3 years - consider breaking into phases');
            }
          }
        }
      }
            
      return oResult;
    },
        
    _validateBudget: function (nBudget, oProjectData) {
      const oResult = { errors: [], warnings: [] };
            
      if (nBudget !== undefined && nBudget !== null && nBudget !== '') {
        if (isNaN(nBudget) || nBudget < 0) {
          oResult.errors.push('Budget must be a positive number');
        } else {
          if (nBudget === 0) {
            oResult.warnings.push('Zero budget may indicate missing budget planning');
          } else if (nBudget > 10000000) { // 10M
            oResult.warnings.push('Budget exceeds 10M - requires executive approval');
          }
                    
          // Check against budget limit if available
          if (oProjectData.budgetLimit && nBudget > oProjectData.budgetLimit) {
            oResult.errors.push(`Budget exceeds approved limit of ${  oProjectData.budgetLimit}`);
          }
        }
      }
            
      return oResult;
    },
        
    _validateCostCenter: function (sCostCenter) {
      const oResult = { errors: [], warnings: [] };
            
      if (sCostCenter && sCostCenter.trim()) {
        // Cost center format validation (example: CC-XXX-### format)
        if (!/^CC-[A-Z]{2,4}-\d{3,4}$/.test(sCostCenter)) {
          oResult.warnings.push('Cost center should follow format: CC-XXX-### (e.g., CC-IT-001)');
        }
      }
            
      return oResult;
    },
        
    _validateCrossFieldRules: function (oProjectData) {
      const oResult = { errors: [], warnings: [] };
            
      // Business rule validations that depend on multiple fields
      if (oProjectData.status === 'DEPLOYED' && (!oProjectData.agents || oProjectData.agents.length === 0)) {
        oResult.errors.push('Cannot deploy project without agents');
      }
            
      if (oProjectData.priority === 'LOW' && oProjectData.budget && oProjectData.budget > 1000000) {
        oResult.warnings.push('High budget project with low priority - review prioritization');
      }
            
      // Date and budget correlation
      if (oProjectData.startDate && oProjectData.endDate && oProjectData.budget) {
        const duration = (new Date(oProjectData.endDate) - new Date(oProjectData.startDate)) / (1000 * 60 * 60 * 24);
        const dailyBudget = oProjectData.budget / duration;
                
        if (dailyBudget > 50000) {
          oResult.warnings.push(`Daily budget rate is very high (${  Math.round(dailyBudget)  }/day)`);
        } else if (dailyBudget < 100) {
          oResult.warnings.push(`Daily budget rate is very low (${  Math.round(dailyBudget)  }/day)`);
        }
      }
            
      return oResult;
    },
        
    _updateValidationDisplay: function (oValidationResult) {
      // Update form field validation states
      Object.keys(oValidationResult.fieldValidation).forEach((sField) => {
        const oFieldResult = oValidationResult.fieldValidation[sField];
        this._updateFieldValidationDisplay(sField, oFieldResult);
      });
            
      // Update validation summary model
      this.getModel('view').setProperty('/validationSummary', oValidationResult.summary);
      this.getModel('view').setProperty('/validationResult', oValidationResult);
            
      // Show/hide validation panel
      this._toggleValidationPanel(oValidationResult.summary.totalErrors > 0 || oValidationResult.summary.totalWarnings > 0);
    },
        
    _updateFieldValidationDisplay: function (sField, oFieldResult) {
      const sControlId = this._getFieldControlId(sField);
      const oControl = this.byId(sControlId);
            
      if (oControl) {
        // Clear previous validation state
        oControl.removeStyleClass('a2a-field-error a2a-field-warning a2a-field-success');
        oControl.setValueState(sap.ui.core.ValueState.None);
        oControl.setValueStateText('');
                
        // Apply new validation state
        if (oFieldResult.errors.length > 0) {
          oControl.addStyleClass('a2a-field-error');
          oControl.setValueState(sap.ui.core.ValueState.Error);
          oControl.setValueStateText(oFieldResult.errors.join('; '));
        } else if (oFieldResult.warnings.length > 0) {
          oControl.addStyleClass('a2a-field-warning');
          oControl.setValueState(sap.ui.core.ValueState.Warning);
          oControl.setValueStateText(oFieldResult.warnings.join('; '));
        } else {
          oControl.addStyleClass('a2a-field-success');
          oControl.setValueState(sap.ui.core.ValueState.Success);
        }
      }
    },
        
    _showValidationSummary: function (oValidationResult) {
      let sMessage = 'Form validation found issues:\n\n';
            
      if (oValidationResult.summary.totalErrors > 0) {
        sMessage += `Errors (${  oValidationResult.summary.totalErrors  }):\n`;
        oValidationResult.errors.forEach((sError, index) => {
          sMessage += `• ${  sError  }\n`;
        });
        sMessage += '\n';
      }
            
      if (oValidationResult.summary.totalWarnings > 0) {
        sMessage += `Warnings (${  oValidationResult.summary.totalWarnings  }):\n`;
        oValidationResult.warnings.forEach((sWarning, index) => {
          sMessage += `• ${  sWarning  }\n`;
        });
      }
            
      MessageBox.error(sMessage, {
        title: 'Validation Results',
        details: 'Please fix the errors before saving. Warnings are advisory but recommended to address.',
        actions: [MessageBox.Action.OK, 'Show Details'],
        emphasizedAction: MessageBox.Action.OK,
        onClose: function(sAction) {
          if (sAction === 'Show Details') {
            this._showDetailedValidationDialog(oValidationResult);
          }
        }.bind(this)
      });
    },
        
    _showDetailedValidationDialog: function (oValidationResult) {
      if (!this._validationDetailsDialog) {
        this._validationDetailsDialog = sap.ui.xmlfragment('a2a.portal.view.fragments.ValidationDetailsDialog', this);
        this.getView().addDependent(this._validationDetailsDialog);
      }
            
      const oModel = new sap.ui.model.json.JSONModel(oValidationResult);
      this._validationDetailsDialog.setModel(oModel, 'validation');
      this._validationDetailsDialog.open();
    },
        
    _toggleValidationPanel: function (bShow) {
      this.getModel('view').setProperty('/showValidationPanel', bShow);
    },
        
    onRealTimeValidation: function (oEvent) {
      // Real-time validation as user types
      const oControl = oEvent.getSource();
      const sValue = oEvent.getParameter('value');
      const sFieldName = this._getFieldNameFromControl(oControl);
            
      if (sFieldName) {
        const oProjectData = this.getModel('project').getData();
        oProjectData[sFieldName] = sValue;
                
        const oFieldResult = this._validateField(sFieldName, sValue, oProjectData);
        this._updateFieldValidationDisplay(sFieldName, oFieldResult);
                
        // Update live validation indicator
        this._updateLiveValidationIndicator();
      }
    },
        
    _getFieldNameFromControl: function (oControl) {
      const sControlId = oControl.getId();
      const mControlFieldMap = {
        'projectNameInput': 'name',
        'projectDescInput': 'description',
        'startDatePicker': 'startDate', 
        'endDatePicker': 'endDate',
        'budgetInput': 'budget',
        'costCenterInput': 'costCenter'
      };
            
      const sLocalId = sControlId.split('--').pop();
      return mControlFieldMap[sLocalId];
    },
        
    _updateLiveValidationIndicator: function () {
      const oProjectData = this.getModel('project').getData();
      const oValidationResult = this._performComprehensiveValidation(oProjectData);
            
      // Update header validation indicator
      this.getModel('view').setProperty('/liveValidation', {
        isValid: oValidationResult.isValid,
        errorCount: oValidationResult.summary.totalErrors,
        warningCount: oValidationResult.summary.totalWarnings,
        validFieldsPercent: Math.round((oValidationResult.summary.validFields / oValidationResult.summary.totalFields) * 100)
      });
    },
        
    onToggleValidationPanel: function () {
      const bShow = this.getModel('view').getProperty('/showValidationPanel');
      this._toggleValidationPanel(!bShow);
    },
        
    onValidateAllFields: function () {
      const oProjectData = this.getModel('project').getData();
      const oValidationResult = this._performComprehensiveValidation(oProjectData);
      this._updateValidationDisplay(oValidationResult);
            
      MessageToast.show(`Validation complete: ${  oValidationResult.summary.validFields  }/${  
        oValidationResult.summary.totalFields  } fields valid`);
    },
        
    onCloseValidationDetails: function () {
      if (this._validationDetailsDialog) {
        this._validationDetailsDialog.close();
      }
    },
        
    _toggleFormControls: function (bEditable) {
      const aEditableControls = [
        'projectNameInput',
        'projectDescInput', 
        'startDatePicker',
        'endDatePicker',
        'budgetInput',
        'costCenterInput'
      ];
            
      aEditableControls.forEach((sControlId) => {
        const oControl = this.byId(sControlId);
        if (oControl && oControl.setEditable) {
          oControl.setEditable(bEditable);
        } else if (oControl && oControl.setEnabled) {
          oControl.setEnabled(bEditable);
        }
      });
    },
        
    _showEditModeIndicators: function (bShow) {
      // Add visual indicators for edit mode
      const oObjectPage = this.byId('objectPageLayout');
      if (oObjectPage) {
        if (bShow) {
          oObjectPage.addStyleClass('a2a-edit-mode');
        } else {
          oObjectPage.removeStyleClass('a2a-edit-mode');
        }
      }
            
      // Update edit mode indicator styles
      this._updateEditModeStyles(bShow);
    },
        
    _updateEditModeStyles: function (bEditMode) {
      const sStyles = `
                .a2a-edit-mode {
                    border-left: 4px solid #007bff !important;
                    background-color: rgba(0, 123, 191, 0.02) !important;
                }
                
                .a2a-edit-mode .sapUxAPObjectPageSectionTitle {
                    color: #007bff !important;
                }
                
                .a2a-edit-indicator {
                    position: absolute !important;
                    top: 10px !important;
                    right: 10px !important;
                    background-color: #28a745 !important;
                    color: white !important;
                    padding: 4px 12px !important;
                    border-radius: 12px !important;
                    font-size: 0.75rem !important;
                    font-weight: bold !important;
                    z-index: 1000 !important;
                    animation: pulseGreen 2s infinite !important;
                }
                
                @keyframes pulseGreen {
                    0% { box-shadow: 0 0 0 0 rgba(40, 167, 69, 0.7); }
                    70% { box-shadow: 0 0 0 10px rgba(40, 167, 69, 0); }
                    100% { box-shadow: 0 0 0 0 rgba(40, 167, 69, 0); }
                }
                
                .a2a-form-control-editable {
                    border: 2px solid #007bff !important;
                    background-color: rgba(0, 123, 191, 0.05) !important;
                }
                
                .a2a-cancel-button {
                    margin-left: 8px !important;
                }
            `;
            
      // Inject or remove edit mode styles
      const oStyleElement = document.getElementById('editModeStyles');
      if (bEditMode && !oStyleElement) {
        oStyleElement = document.createElement('style');
        oStyleElement.id = 'editModeStyles';
        oStyleElement.textContent = sStyles;
        document.head.appendChild(oStyleElement);
                
        // Add edit indicator to header
        this._addEditIndicator();
      } else if (!bEditMode && oStyleElement) {
        oStyleElement.remove();
        this._removeEditIndicator();
      }
    },
        
    _addEditIndicator: function () {
      const $headerContent = $('.sapUxAPObjectPageHeaderContent').first();
      if ($headerContent.length && !$headerContent.find('.a2a-edit-indicator').length) {
        const $indicator = $('<div class="a2a-edit-indicator">EDIT MODE</div>');
        $headerContent.append($indicator);
      }
    },
        
    _removeEditIndicator: function () {
      $('.a2a-edit-indicator').remove();
    },
        
    onCancelEdit: function () {
      if (!this._originalData) {
        MessageToast.show('No changes to cancel');
        return;
      }
            
      // Calculate current changes
      const oCurrentData = this.getModel('project').getData();
      const oChanges = this._calculateChanges(this._originalData, oCurrentData);
            
      if (Object.keys(oChanges).length === 0) {
        // No changes detected, just exit edit mode
        this._exitEditModeCleanly();
        MessageToast.show('No changes to cancel - exiting edit mode');
        return;
      }
            
      // Show advanced cancel dialog with change details
      this._showAdvancedCancelDialog(oChanges);
    },
        
    _showAdvancedCancelDialog: function (oChanges) {
      const sChangesSummary = this._formatChangesForCancelDialog(oChanges);
      const sDialogContent = `You have unsaved changes:\n\n${  sChangesSummary  }\n\nWhat would you like to do?`;
            
      MessageBox.show(sDialogContent, {
        title: 'Unsaved Changes Detected',
        actions: [
          new sap.m.Button({
            text: 'Discard All Changes',
            type: 'Reject',
            press: function () {
              this._discardAllChanges();
              MessageBox.close();
            }.bind(this)
          }),
          new sap.m.Button({
            text: 'Review Changes',
            type: 'Default',
            press: function () {
              this._showChangeReviewDialog(oChanges);
              MessageBox.close();
            }.bind(this)
          }),
          new sap.m.Button({
            text: 'Save & Exit',
            type: 'Emphasized',
            press: function () {
              this._saveAndExitEditMode();
              MessageBox.close();
            }.bind(this)
          }),
          new sap.m.Button({
            text: 'Continue Editing',
            type: 'Default',
            press: function () {
              MessageBox.close();
            }
          })
        ],
        emphasizedAction: 'Save & Exit',
        initialFocus: 'Continue Editing'
      });
    },
        
    _formatChangesForCancelDialog: function (oChanges) {
      const aChangeDescriptions = [];
      Object.keys(oChanges).forEach((sField) => {
        const oChange = oChanges[sField];
        const sFromValue = oChange.from ? String(oChange.from) : '(empty)';
        const sToValue = oChange.to ? String(oChange.to) : '(empty)';
                
        // Truncate long values for dialog display
        if (sFromValue.length > 50) {
          sFromValue = `${sFromValue.substring(0, 47)  }...`;
        }
        if (sToValue.length > 50) {
          sToValue = `${sToValue.substring(0, 47)  }...`;
        }
                
        aChangeDescriptions.push(`• ${  this._getFieldDisplayName(sField)  }: ${  sFromValue  } → ${  sToValue}`);
      });
            
      return aChangeDescriptions.join('\n');
    },
        
    _getFieldDisplayName: function (sField) {
      const mFieldNames = {
        'name': 'Project Name',
        'description': 'Description', 
        'startDate': 'Start Date',
        'endDate': 'End Date',
        'budget': 'Budget',
        'costCenter': 'Cost Center',
        'priority': 'Priority',
        'status': 'Status'
      };
      return mFieldNames[sField] || sField;
    },
        
    _showChangeReviewDialog: function (oChanges) {
      const oModel = new sap.ui.model.json.JSONModel({
        changes: Object.keys(oChanges).map((sField) => {
          return {
            field: sField,
            fieldDisplay: this._getFieldDisplayName(sField),
            from: oChanges[sField].from || '(empty)',
            to: oChanges[sField].to || '(empty)',
            selected: true // All changes selected for discard by default
          };
        })
      });
            
      if (!this._changeReviewDialog) {
        this._changeReviewDialog = sap.ui.xmlfragment('a2a.portal.view.fragments.ChangeReviewDialog', this);
        this.getView().addDependent(this._changeReviewDialog);
      }
            
      this._changeReviewDialog.setModel(oModel, 'changes');
      this._changeReviewDialog.open();
    },
        
    onSelectiveCancel: function () {
      const oChangesModel = this._changeReviewDialog.getModel('changes');
      const aChanges = oChangesModel.getProperty('/changes');
      const aSelectedFields = aChanges.filter((oChange) => {
        return oChange.selected;
      }).map((oChange) => {
        return oChange.field;
      });
            
      if (aSelectedFields.length === 0) {
        MessageToast.show('No changes selected to discard');
        return;
      }
            
      this._discardSelectedChanges(aSelectedFields);
      this._changeReviewDialog.close();
    },
        
    onCancelChangeReview: function () {
      this._changeReviewDialog.close();
    },
        
    _discardSelectedChanges: function (aFieldsToRevert) {
      const oCurrentData = this.getModel('project').getData();
      const oOriginalData = this._originalData;
            
      // Revert only selected fields
      aFieldsToRevert.forEach((sField) => {
        if (oOriginalData.hasOwnProperty(sField)) {
          oCurrentData[sField] = oOriginalData[sField];
        }
      });
            
      // Update the model
      this.getModel('project').setData(oCurrentData);
            
      // Check if any changes remain
      const oRemainingChanges = this._calculateChanges(oOriginalData, oCurrentData);
            
      if (Object.keys(oRemainingChanges).length === 0) {
        // No changes remain, exit edit mode
        this._exitEditModeCleanly();
        MessageToast.show('All changes discarded - exiting edit mode');
      } else {
        MessageToast.show(`${aFieldsToRevert.length  } field(s) reverted - remaining changes preserved`);
      }
    },
        
    _discardAllChanges: function () {
      // Show discard progress for better UX
      this.getModel('view').setProperty('/busy', true);
            
      setTimeout(() => {
        // Restore original data
        this.getModel('project').setData(jQuery.extend(true, {}, this._originalData));
                
        // Exit edit mode
        this._exitEditModeCleanly();
                
        this.getModel('view').setProperty('/busy', false);
        MessageToast.show('All changes discarded - edit cancelled');
                
        // Log cancellation for audit
        this._logCancellation('ALL_CHANGES');
      }, 300);
    },
        
    _saveAndExitEditMode: function () {
      // Attempt to save changes before exiting
      const bSaveSuccessful = this._saveChanges();
            
      // Note: _saveChanges handles the exit from edit mode on success
      if (!bSaveSuccessful) {
        MessageToast.show('Save failed - remaining in edit mode');
      }
    },
        
    _exitEditModeCleanly: function () {
      // Clean exit from edit mode
      this.getModel('view').setProperty('/editMode', false);
      this._disableEditMode();
            
      // Clear original data backup
      delete this._originalData;
            
      // Remove any unsaved change indicators
      this._clearChangeIndicators();
    },
        
    _clearChangeIndicators: function () {
      // Remove any visual indicators of unsaved changes
      const aFormControls = ['projectNameInput', 'projectDescInput', 'startDatePicker', 'endDatePicker', 'budgetInput', 'costCenterInput'];
            
      aFormControls.forEach((sControlId) => {
        const oControl = this.byId(sControlId);
        if (oControl) {
          oControl.removeStyleClass('a2a-field-changed');
          oControl.removeStyleClass('a2a-field-unsaved');
        }
      });
    },
        
    _logCancellation: function (sType, aFields) {
      // Log cancellation for audit purposes
      const oCancelLog = {
        timestamp: new Date().toISOString(),
        projectId: this._projectId,
        userId: this.getModel('view').getProperty('/fieldEditability/userRole'),
        cancelType: sType,
        fieldsAffected: aFields || [],
        sessionId: this._generateSessionId()
      };
            
      // In real implementation, this would send to audit service
      // eslint-disable-next-line no-console
      // eslint-disable-next-line no-console
      console.log('Edit cancellation logged:', oCancelLog);
    },
        
    _generateSessionId: function () {
      return `session_${  Date.now()  }_${  Math.random().toString(36).substr(2, 9)}`;
    },
        
    onPreventNavigation: function () {
      // Prevent navigation when there are unsaved changes
      if (this._hasUnsavedChanges()) {
        MessageBox.confirm(
          'You have unsaved changes. Are you sure you want to leave this page?', {
            title: 'Unsaved Changes',
            actions: [MessageBox.Action.YES, MessageBox.Action.NO],
            emphasizedAction: MessageBox.Action.NO,
            onClose: function (sAction) {
              if (sAction === MessageBox.Action.YES) {
                // Allow navigation - user confirmed
                this._allowNavigation();
              }
              // Otherwise prevent navigation
            }.bind(this)
          }
        );
        return false; // Prevent navigation initially
      }
      return true; // Allow navigation
    },
        
    _hasUnsavedChanges: function () {
      if (!this._originalData) {
        return false;
      }
            
      const oCurrentData = this.getModel('project').getData();
      const oChanges = this._calculateChanges(this._originalData, oCurrentData);
      return Object.keys(oChanges).length > 0;
    },
        
    _allowNavigation: function () {
      // Clean up before allowing navigation
      this._discardAllChanges();
    },
        
    onAutoSaveDraft: function () {
      if (!this._hasUnsavedChanges()) {
        return;
      }
            
      // Auto-save as draft (not full save)
      const oCurrentData = this.getModel('project').getData();
      const oDraftData = jQuery.extend(true, {}, oCurrentData);
      oDraftData.isDraft = true;
      oDraftData.draftTimestamp = new Date().toISOString();
            
      // Save to local storage as backup
      try {
        localStorage.setItem(`a2a_project_draft_${  this._projectId}`, JSON.stringify(oDraftData));
        MessageToast.show('Draft auto-saved locally');
      } catch (e) {
        console.warn('Could not save draft to local storage:', e);
      }
    },
        
    onRestoreFromDraft: function () {
      try {
        const sDraftData = localStorage.getItem(`a2a_project_draft_${  this._projectId}`);
        if (sDraftData) {
          const oDraftData = JSON.parse(sDraftData);
                    
          MessageBox.confirm(
            `A draft was found from ${  new Date(oDraftData.draftTimestamp).toLocaleString()  }. Restore it?`, {
              title: 'Draft Available',
              onClose: function (sAction) {
                if (sAction === MessageBox.Action.OK) {
                  delete oDraftData.isDraft;
                  delete oDraftData.draftTimestamp;
                  this.getModel('project').setData(oDraftData);
                  MessageToast.show('Draft restored');
                }
              }.bind(this)
            }
          );
        } else {
          MessageToast.show('No draft available');
        }
      } catch (e) {
        MessageBox.error(`Could not restore draft: ${  e.message}`);
      }
    },
        
    onSelectAllChanges: function () {
      const oChangesModel = this._changeReviewDialog.getModel('changes');
      const aChanges = oChangesModel.getProperty('/changes');
            
      aChanges.forEach((oChange) => {
        oChange.selected = true;
      });
            
      oChangesModel.setProperty('/changes', aChanges);
    },
        
    onDeselectAllChanges: function () {
      const oChangesModel = this._changeReviewDialog.getModel('changes');
      const aChanges = oChangesModel.getProperty('/changes');
            
      aChanges.forEach((oChange) => {
        oChange.selected = false;
      });
            
      oChangesModel.setProperty('/changes', aChanges);
    },

    _saveChanges: function () {
      const oProjectData = this.getModel('project').getData();
      const oOriginalData = this._originalData;
            
      // Perform pre-save validation
      if (!this._preSaveValidation(oProjectData)) {
        return false;
      }
            
      // Calculate and log changes
      const oChanges = this._calculateChanges(oOriginalData, oProjectData);
      if (Object.keys(oChanges).length === 0) {
        MessageToast.show('No changes detected to save');
        return false;
      }
            
      // Show advanced loading indicator
      this._showSaveProgress('Preparing to save changes...', 0);
            
      // Perform optimistic locking check
      this._checkForConflicts(oProjectData).then((bConflictExists) => {
        if (bConflictExists) {
          this._handleSaveConflict();
          return;
        }
                
        // Proceed with save
        this._performSave(oProjectData, oChanges);
      });
    },
        
    _preSaveValidation: function (oProjectData) {
      const bValid = true;
      const aErrors = [];
      const aWarnings = [];
            
      // Enhanced validation beyond basic form validation
      if (oProjectData.startDate && oProjectData.endDate) {
        const startDate = new Date(oProjectData.startDate);
        const endDate = new Date(oProjectData.endDate);
        const now = new Date();
                
        if (startDate < now && oProjectData.status === 'DRAFT') {
          aWarnings.push('Start date is in the past for a draft project');
        }
                
        const daysDiff = (endDate - startDate) / (1000 * 60 * 60 * 24);
        if (daysDiff < 7) {
          aErrors.push('Project duration must be at least 7 days');
          bValid = false;
        }
                
        if (daysDiff > 365 * 3) {
          aWarnings.push('Project duration exceeds 3 years - consider breaking into phases');
        }
      }
            
      // Budget validation
      if (oProjectData.budget) {
        if (oProjectData.budget > 1000000) {
          aWarnings.push('Budget exceeds $1M - requires executive approval');
        }
                
        if (oProjectData.budgetLimit && oProjectData.budget > oProjectData.budgetLimit) {
          aErrors.push(`Budget exceeds approved limit of ${  oProjectData.budgetLimit}`);
          bValid = false;
        }
      }
            
      // Business rule validation
      if (oProjectData.status === 'DEPLOYED' && (!oProjectData.agents || oProjectData.agents.length === 0)) {
        aErrors.push('Cannot deploy project without agents');
        bValid = false;
      }
            
      // Show warnings if any
      if (aWarnings.length > 0) {
        const sWarningMessage = `Warnings:\n${  aWarnings.join('\n')  }\n\nContinue saving?`;
        return new Promise((resolve) => {
          MessageBox.confirm(sWarningMessage, {
            title: 'Save Warnings',
            onClose: function(sAction) {
              resolve(sAction === MessageBox.Action.OK);
            }
          });
        });
      }
            
      // Show errors if any
      if (!bValid) {
        MessageBox.error(`Please fix the following errors:\n\n${  aErrors.join('\n')}`, {
          title: 'Validation Errors'
        });
      }
            
      return bValid;
    },
        
    _calculateChanges: function (oOriginal, oCurrent) {
      const oChanges = {};
      const aTrackableFields = ['name', 'description', 'startDate', 'endDate', 'budget', 'costCenter', 'priority', 'status'];
            
      aTrackableFields.forEach((sField) => {
        const originalValue = oOriginal ? oOriginal[sField] : null;
        const currentValue = oCurrent[sField];
                
        if (originalValue !== currentValue) {
          oChanges[sField] = {
            from: originalValue,
            to: currentValue,
            timestamp: new Date().toISOString(),
            user: this.getModel('view').getProperty('/fieldEditability/userRole')
          };
        }
      });
            
      return oChanges;
    },
        
    _checkForConflicts: function (oProjectData) {
      return new Promise((resolve) => {
        // Simulate checking for concurrent edits
        jQuery.ajax({
          url: `/api/projects/${  this._projectId  }/version`,
          method: 'GET',
          success: function (versionData) {
            const bConflict = versionData.lastModified !== oProjectData.lastModified;
            resolve(bConflict);
          }.bind(this),
          error: function () {
            // No conflict check possible - assume no conflict
            resolve(false);
          }
        });
      });
    },
        
    _handleSaveConflict: function () {
      MessageBox.warning(
        'This project has been modified by another user since you started editing. Your changes may conflict with theirs.', {
          title: 'Concurrent Edit Detected',
          actions: [MessageBox.Action.YES, MessageBox.Action.NO, MessageBox.Action.CANCEL],
          emphasizedAction: MessageBox.Action.YES,
          details: "Choose 'Yes' to overwrite their changes, 'No' to reload and lose your changes, or 'Cancel' to continue editing.",
          onClose: function (sAction) {
            if (sAction === MessageBox.Action.YES) {
              // Force save - overwrite changes
              this._performSave(this.getModel('project').getData(), {}, true);
            } else if (sAction === MessageBox.Action.NO) {
              // Reload data and lose changes
              this._loadProjectDetails(this._projectId);
              this._disableEditMode();
              MessageToast.show('Project reloaded - your changes were discarded');
            }
            // Cancel - do nothing, stay in edit mode
          }.bind(this)
        }
      );
    },
        
    _showSaveProgress: function (sMessage, iProgress) {
      this.getModel('view').setProperty('/busy', true);
      this.getModel('view').setProperty('/saveProgress', {
        message: sMessage,
        progress: iProgress
      });
    },
        
    _performSave: function (oProjectData, oChanges, bForceOverwrite) {
      const that = this;
      const iTotalSteps = 5;
      const iCurrentStep = 0;
            
      // Step 1: Prepare data
      this._showSaveProgress('Preparing data for save...', Math.round((++iCurrentStep / iTotalSteps) * 100));
            
      setTimeout(() => {
        // Step 2: Validate server-side
        that._showSaveProgress('Server-side validation...', Math.round((++iCurrentStep / iTotalSteps) * 100));
                
        setTimeout(() => {
          // Step 3: Save to database
          that._showSaveProgress('Saving to database...', Math.round((++iCurrentStep / iTotalSteps) * 100));
                    
          // Perform actual save
          jQuery.ajax({
            url: `/api/projects/${  that._projectId}`,
            method: 'PUT',
            headers: {
              'Content-Type': 'application/json',
              'If-Unmodified-Since': bForceOverwrite ? null : oProjectData.lastModified,
              'X-Change-Summary': JSON.stringify(Object.keys(oChanges))
            },
            data: JSON.stringify({
              project: oProjectData,
              changes: oChanges,
              metadata: {
                saveTimestamp: new Date().toISOString(),
                userRole: that.getModel('view').getProperty('/fieldEditability/userRole'),
                clientVersion: '1.0.0'
              }
            }),
            success: function (response) {
              // Step 4: Update caches
              that._showSaveProgress('Updating caches...', Math.round((++iCurrentStep / iTotalSteps) * 100));
                            
              setTimeout(() => {
                // Step 5: Finalize
                that._showSaveProgress('Finalizing save...', 100);
                                
                setTimeout(() => {
                  that._completeSaveSuccess(response, oChanges);
                }, 300);
              }, 200);
            },
            error: function (xhr, status, error) {
              that._handleSaveError(xhr, status, error, oChanges);
            }
          });
        }, 400);
      }, 300);
    },
        
    _completeSaveSuccess: function (response, oChanges) {
      // Hide loading
      this.getModel('view').setProperty('/busy', false);
      this.getModel('view').setProperty('/saveProgress', null);
            
      // Update project data with server response
      if (response && response.project) {
        this.getModel('project').setData(response.project);
      }
            
      // Update last modified timestamp
      this.getModel('project').setProperty('/lastModified', new Date());
            
      // Log changes for audit trail
      this._logChangeHistory(oChanges);
            
      // Show success message with change summary
      const sChangesSummary = this._formatChangesSummary(oChanges);
      MessageToast.show(`Changes saved successfully!\n${  sChangesSummary}`);
            
      // Refresh field editability
      this._updateFieldEditability();
            
      // Trigger save event for external listeners
      this.fireEvent('projectSaved', {
        projectId: this._projectId,
        changes: oChanges,
        timestamp: new Date()
      });
    },
        
    _handleSaveError: function (xhr, status, error, oChanges) {
      this.getModel('view').setProperty('/busy', false);
      this.getModel('view').setProperty('/saveProgress', null);
            
      const sErrorMessage = 'Failed to save changes';
      const sErrorDetails = '';
            
      // Parse error response
      try {
        const oErrorResponse = JSON.parse(xhr.responseText);
        sErrorMessage = oErrorResponse.message || sErrorMessage;
        sErrorDetails = oErrorResponse.details || '';
      } catch (e) {
        sErrorDetails = `HTTP ${  xhr.status  }: ${  error}`;
      }
            
      // Handle specific error types
      if (xhr.status === 409) {
        // Conflict - concurrent modification
        this._handleSaveConflict();
        return;
      } else if (xhr.status === 422) {
        // Validation error
        MessageBox.error(sErrorMessage, {
          title: 'Validation Error',
          details: sErrorDetails
        });
      } else if (xhr.status === 403) {
        // Permission error
        MessageBox.error("You don't have permission to save these changes", {
          title: 'Permission Denied',
          details: sErrorDetails
        });
      } else {
        // Generic error with retry option
        MessageBox.error(sErrorMessage, {
          title: 'Save Failed',
          details: sErrorDetails,
          actions: [MessageBox.Action.RETRY, MessageBox.Action.CANCEL],
          emphasizedAction: MessageBox.Action.RETRY,
          onClose: function (sAction) {
            if (sAction === MessageBox.Action.RETRY) {
              this._performSave(this.getModel('project').getData(), oChanges);
            }
          }.bind(this)
        });
      }
            
      // Log error for debugging
      console.error('Save error:', {
        status: xhr.status,
        error: error,
        response: xhr.responseText,
        changes: oChanges
      });
    },
        
    _logChangeHistory: function (oChanges) {
      // Store change history for audit purposes
      const aChangeHistory = this.getModel('view').getProperty('/changeHistory') || [];
            
      Object.keys(oChanges).forEach((sField) => {
        aChangeHistory.push({
          field: sField,
          change: oChanges[sField],
          savedAt: new Date().toISOString()
        });
      });
            
      // Keep only last 50 changes
      if (aChangeHistory.length > 50) {
        aChangeHistory = aChangeHistory.slice(-50);
      }
            
      this.getModel('view').setProperty('/changeHistory', aChangeHistory);
    },
        
    _formatChangesSummary: function (oChanges) {
      const aChangeSummary = [];
      Object.keys(oChanges).forEach((sField) => {
        const oChange = oChanges[sField];
        aChangeSummary.push(`${sField  }: ${  oChange.from || 'empty'  } → ${  oChange.to || 'empty'}`);
      });
            
      return aChangeSummary.length > 0 ? `\nChanges: ${  aChangeSummary.join(', ')}` : '';
    },
        
    onViewChangeHistory: function () {
      const aChangeHistory = this.getModel('view').getProperty('/changeHistory') || [];
            
      if (aChangeHistory.length === 0) {
        MessageBox.information('No change history available for this session.', {
          title: 'Change History'
        });
        return;
      }
            
      const sHistoryText = aChangeHistory.map((oChange, index) => {
        return `${index + 1  }. ${  oChange.field  }: ${  
          oChange.change.from || 'empty'  } → ${  oChange.change.to || 'empty' 
        } (saved at ${  new Date(oChange.savedAt).toLocaleTimeString()  })`;
      }).join('\n');
            
      MessageBox.information(sHistoryText, {
        title: `Change History (${  aChangeHistory.length  } changes)`,
        contentWidth: '30em'
      });
    },
        
    onForceSave: function () {
      MessageBox.confirm(
        'Force save will overwrite any concurrent changes made by other users. Are you sure?', {
          title: 'Force Save Confirmation',
          onClose: function (sAction) {
            if (sAction === MessageBox.Action.OK) {
              this._performSave(this.getModel('project').getData(), 
                this._calculateChanges(this._originalData, this.getModel('project').getData()), 
                true);
            }
          }.bind(this)
        }
      );
    },
        
    _initializeFieldEditability: function () {
      // Initialize field editability system
      this._fieldEditabilityConfig = this.getModel('view').getProperty('/fieldEditability');
            
      // Load user role from session/backend
      this._loadUserRole();
            
      // Setup initial field permissions
      this._updateFieldEditability();
            
      // Add field-level event handlers
      this._setupFieldEditabilityHandlers();
    },
        
    _loadUserRole: function () {
      // In real implementation, this would call user service
      jQuery.ajax({
        url: '/api/user/profile',
        method: 'GET',
        success: function (data) {
          this.getModel('view').setProperty('/fieldEditability/userRole', data.role || 'DEVELOPER');
          this._updateFieldEditability();
        }.bind(this),
        error: function () {
          // Use mock role for demo
          this.getModel('view').setProperty('/fieldEditability/userRole', 'PROJECT_MANAGER');
          this._updateFieldEditability();
        }.bind(this)
      });
    },
        
    _updateFieldEditability: function () {
      const oEditabilityConfig = this.getModel('view').getProperty('/fieldEditability');
      const sUserRole = oEditabilityConfig.userRole;
      const sProjectStatus = this.getModel('project').getProperty('/status') || 'ACTIVE';
            
      // Get role-based permissions
      const aRolePermissions = oEditabilityConfig.rolePermissions[sUserRole] || [];
            
      // Get status-based restrictions
      const aStatusRestrictions = oEditabilityConfig.statusRestrictions[sProjectStatus];
      if (aStatusRestrictions === 'ALL') {
        aStatusRestrictions = aRolePermissions;
      } else if (!aStatusRestrictions) {
        aStatusRestrictions = [];
      }
            
      // Calculate final editable fields (intersection of role and status permissions)
      const aEditableFields = aRolePermissions.filter((field) => {
        return aStatusRestrictions.includes(field);
      });
            
      // Update field editability state
      const oEditableFields = {};
      const oReadOnlyFields = {};
            
      ['name', 'description', 'startDate', 'endDate', 'budget', 'costCenter', 'priority', 'status'].forEach((field) => {
        if (aEditableFields.includes(field)) {
          oEditableFields[field] = true;
          oReadOnlyFields[field] = false;
        } else {
          oEditableFields[field] = false;
          oReadOnlyFields[field] = true;
        }
      });
            
      // Update model
      this.getModel('view').setProperty('/fieldEditability/editableFields', oEditableFields);
      this.getModel('view').setProperty('/fieldEditability/readOnlyFields', oReadOnlyFields);
      this.getModel('view').setProperty('/fieldEditability/projectStatus', sProjectStatus);
            
      // Update form controls
      this._applyFieldEditabilityToControls();
            
      // Update field styling
      this._updateFieldEditabilityStyles();
    },
        
    _applyFieldEditabilityToControls: function () {
      const oEditableFields = this.getModel('view').getProperty('/fieldEditability/editableFields');
      const bEditMode = this.getModel('view').getProperty('/editMode');
            
      // Apply editability to each field
      Object.keys(oEditableFields).forEach((sField) => {
        const bFieldEditable = oEditableFields[sField];
        const sControlId = this._getFieldControlId(sField);
        const oControl = this.byId(sControlId);
                
        if (oControl) {
          if (oControl.setEditable) {
            oControl.setEditable(bEditMode && bFieldEditable);
          } else if (oControl.setEnabled) {
            oControl.setEnabled(bEditMode && bFieldEditable);
          }
                    
          // Add visual styling for read-only fields
          if (bEditMode && !bFieldEditable) {
            oControl.addStyleClass('a2a-field-readonly');
          } else {
            oControl.removeStyleClass('a2a-field-readonly');
          }
        }
      });
    },
        
    _getFieldControlId: function (sField) {
      const mFieldControlMap = {
        'name': 'projectNameInput',
        'description': 'projectDescInput', 
        'startDate': 'startDatePicker',
        'endDate': 'endDatePicker',
        'budget': 'budgetInput',
        'costCenter': 'costCenterInput',
        'priority': 'prioritySelect',
        'status': 'statusSelect'
      };
      return mFieldControlMap[sField];
    },
        
    _setupFieldEditabilityHandlers: function () {
      // Add tooltip handlers for read-only fields
      this._attachReadOnlyTooltips();
            
      // Add field focus handlers for editability feedback
      this._attachFieldFocusHandlers();
    },
        
    _attachReadOnlyTooltips: function () {
      const that = this;
      const aFieldIds = ['projectNameInput', 'projectDescInput', 'startDatePicker', 'endDatePicker', 'budgetInput', 'costCenterInput'];
            
      aFieldIds.forEach((sControlId) => {
        const oControl = that.byId(sControlId);
        if (oControl) {
          oControl.attachBrowserEvent('mouseenter', () => {
            const oEditableFields = that.getModel('view').getProperty('/fieldEditability/editableFields');
            const sField = that._getFieldFromControlId(sControlId);
            const bEditMode = that.getModel('view').getProperty('/editMode');
                        
            if (bEditMode && !oEditableFields[sField]) {
              const sReason = that._getReadOnlyReason(sField);
              oControl.setTooltip(`Read-only: ${  sReason}`);
            } else {
              oControl.setTooltip(null);
            }
          });
        }
      });
    },
        
    _getFieldFromControlId: function (sControlId) {
      const mControlFieldMap = {
        'projectNameInput': 'name',
        'projectDescInput': 'description',
        'startDatePicker': 'startDate', 
        'endDatePicker': 'endDate',
        'budgetInput': 'budget',
        'costCenterInput': 'costCenter'
      };
      return mControlFieldMap[sControlId];
    },
        
    _getReadOnlyReason: function (sField) {
      const oConfig = this.getModel('view').getProperty('/fieldEditability');
      const sUserRole = oConfig.userRole;
      const sProjectStatus = oConfig.projectStatus;
            
      const aRolePermissions = oConfig.rolePermissions[sUserRole] || [];
      const aStatusRestrictions = oConfig.statusRestrictions[sProjectStatus];
            
      if (!aRolePermissions.includes(sField)) {
        return `Insufficient permissions for role '${  sUserRole  }'`;
      } else if (aStatusRestrictions !== 'ALL' && !aStatusRestrictions.includes(sField)) {
        return `Field locked due to project status '${  sProjectStatus  }'`;
      }
      return 'Field is read-only';
    },
        
    _attachFieldFocusHandlers: function () {
      const that = this;
      const aFieldIds = ['projectNameInput', 'projectDescInput', 'startDatePicker', 'endDatePicker', 'budgetInput', 'costCenterInput'];
            
      aFieldIds.forEach((sControlId) => {
        const oControl = that.byId(sControlId);
        if (oControl && oControl.attachBrowserEvent) {
          oControl.attachBrowserEvent('focus', () => {
            that._onFieldFocus(sControlId);
          });
                    
          oControl.attachBrowserEvent('blur', () => {
            that._onFieldBlur(sControlId);
          });
        }
      });
    },
        
    _onFieldFocus: function (sControlId) {
      const oControl = this.byId(sControlId);
      const sField = this._getFieldFromControlId(sControlId);
      const oEditableFields = this.getModel('view').getProperty('/fieldEditability/editableFields');
      const bEditMode = this.getModel('view').getProperty('/editMode');
            
      if (bEditMode && oEditableFields[sField]) {
        oControl.addStyleClass('a2a-field-focused');
      } else if (bEditMode && !oEditableFields[sField]) {
        oControl.addStyleClass('a2a-field-readonly-focus');
        MessageToast.show(`Field '${  sField  }' is read-only: ${  this._getReadOnlyReason(sField)}`);
      }
    },
        
    _onFieldBlur: function (sControlId) {
      const oControl = this.byId(sControlId);
      oControl.removeStyleClass('a2a-field-focused');
      oControl.removeStyleClass('a2a-field-readonly-focus');
    },
        
    _updateFieldEditabilityStyles: function () {
      const sStyles = `
                .a2a-field-readonly {
                    background-color: #f5f5f5 !important;
                    border-color: #d0d0d0 !important;
                    cursor: not-allowed !important;
                    opacity: 0.7 !important;
                }
                
                .a2a-field-readonly:hover {
                    background-color: #f0f0f0 !important;
                    border-color: #c0c0c0 !important;
                }
                
                .a2a-field-readonly-focus {
                    border-color: #dc3545 !important;
                    box-shadow: 0 0 0 0.2rem rgba(220, 53, 69, 0.25) !important;
                }
                
                .a2a-field-focused {
                    border-color: #007bff !important;
                    box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25) !important;
                }
                
                .a2a-field-permission-indicator {
                    display: inline-block !important;
                    width: 8px !important;
                    height: 8px !important;
                    border-radius: 50% !important;
                    margin-left: 8px !important;
                    vertical-align: middle !important;
                }
                
                .a2a-field-permission-editable {
                    background-color: #28a745 !important;
                }
                
                .a2a-field-permission-readonly {
                    background-color: #dc3545 !important;
                }
            `;
            
      // Inject styles
      if (!document.getElementById('fieldEditabilityStyles')) {
        const oStyleElement = document.createElement('style');
        oStyleElement.id = 'fieldEditabilityStyles';
        oStyleElement.textContent = sStyles;
        document.head.appendChild(oStyleElement);
      }
    },
        
    onChangeUserRole: function (oEvent) {
      const sNewRole = oEvent.getParameter('selectedItem').getKey();
      this.getModel('view').setProperty('/fieldEditability/userRole', sNewRole);
      this._updateFieldEditability();
      MessageToast.show(`User role changed to: ${  sNewRole}`);
    },
        
    onTestFieldEditability: function () {
      const oConfig = this.getModel('view').getProperty('/fieldEditability');
      const sMessage = `Current Role: ${  oConfig.userRole  }\n`;
      sMessage += `Project Status: ${  oConfig.projectStatus  }\n`;
      sMessage += `Editable Fields: ${  Object.keys(oConfig.editableFields).filter((field) => {
        return oConfig.editableFields[field];
      }).join(', ')}`;
            
      MessageBox.information(sMessage, {
        title: 'Field Editability Status'
      });
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

    _initializeAnchorBar: function () {
      // Initialize anchor bar configuration
      this._anchorBarConfig = {
        autoScrollToSection: true,
        highlightCurrentSection: true,
        smoothScrollBehavior: true,
        scrollOffset: 60, // Offset for fixed header
        scrollThreshold: 50, // Minimum scroll distance for section change
        scrollDebounce: 100, // Debounce scroll events (ms)
        enableScrollMemory: true, // Remember scroll positions
        enableScrollSpy: true // Track scroll position for highlighting
      };
            
      // Initialize scroll tracking
      this._scrollPositions = {};
      this._scrollDirection = null;
      this._lastScrollTop = 0;
            
      // Setup anchor bar items
      this._setupAnchorBarItems();
            
      // Setup scroll listener for section highlighting
      this._setupScrollListener();
            
      // Setup advanced scroll handling
      this._setupAdvancedScrolling();
            
      // Setup keyboard navigation
      this._setupAnchorKeyboardNavigation();
    },
        
    _setupAnchorBarItems: function () {
      const aAnchorItems = [
        {
          key: 'general',
          title: 'General Information',
          sectionId: 'generalSection',
          icon: 'sap-icon://hint',
          visible: true
        },
        {
          key: 'agents',
          title: 'Agents',
          sectionId: 'agentsSection', 
          icon: 'sap-icon://person-placeholder',
          visible: true,
          badge: 0 // Will be updated with agent count
        },
        {
          key: 'workflows',
          title: 'Workflows',
          sectionId: 'workflowsSection',
          icon: 'sap-icon://process',
          visible: true
        },
        {
          key: 'team',
          title: 'Team Members',
          sectionId: 'teamSection',
          icon: 'sap-icon://group',
          visible: true
        },
        {
          key: 'metrics',
          title: 'Metrics & Analytics',
          sectionId: 'metricsSection',
          icon: 'sap-icon://line-chart',
          visible: true
        },
        {
          key: 'activity',
          title: 'Recent Activity',
          sectionId: 'activitySection',
          icon: 'sap-icon://history',
          visible: true
        }
      ];
            
      this.getModel('view').setProperty('/anchorBarItems', aAnchorItems);
    },
        
    _setupScrollListener: function () {
      // Add scroll listener to highlight current section
      const oObjectPageLayout = this.byId('objectPageLayout');
      if (oObjectPageLayout) {
        oObjectPageLayout.attachNavigate(this._onSectionNavigate.bind(this));
        oObjectPageLayout.attachSectionChange(this._onSectionChange.bind(this));
      }
    },
        
    _setupAnchorKeyboardNavigation: function () {
      // Setup keyboard shortcuts for anchor navigation and scrolling
      const oView = this.getView();
      oView.addEventDelegate({
        onkeydown: function (oEvent) {
          // Ctrl+1-6 for section navigation
          if (oEvent.ctrlKey && oEvent.which >= 49 && oEvent.which <= 54) {
            const iIndex = oEvent.which - 49;
            this._navigateToSectionByIndex(iIndex);
            oEvent.preventDefault();
          }
          // Alt+A for anchor bar focus
          else if (oEvent.altKey && oEvent.which === 65) {
            this._focusAnchorBar();
            oEvent.preventDefault();
          }
          // Home key - scroll to top
          else if (oEvent.which === 36 && oEvent.ctrlKey) {
            this.onScrollToTop();
            oEvent.preventDefault();
          }
          // End key - scroll to bottom
          else if (oEvent.which === 35 && oEvent.ctrlKey) {
            this.onScrollToBottom();
            oEvent.preventDefault();
          }
          // Page Up - scroll up by page
          else if (oEvent.which === 33) {
            this.onScrollByPage(false);
            oEvent.preventDefault();
          }
          // Page Down - scroll down by page
          else if (oEvent.which === 34) {
            this.onScrollByPage(true);
            oEvent.preventDefault();
          }
          // Alt+Up - previous section
          else if (oEvent.altKey && oEvent.which === 38) {
            this._navigateToPreviousSection();
            oEvent.preventDefault();
          }
          // Alt+Down - next section
          else if (oEvent.altKey && oEvent.which === 40) {
            this._navigateToNextSection();
            oEvent.preventDefault();
          }
        }.bind(this)
      });
    },
        
    _onSectionNavigate: function (oEvent) {
      const sTargetId = oEvent.getParameter('targetId');
      this._updateCurrentSection(sTargetId);
    },
        
    _onSectionChange: function (oEvent) {
      const oSection = oEvent.getParameter('section');
      if (oSection) {
        this._updateCurrentSection(oSection.getId());
      }
    },
        
    _updateCurrentSection: function (sSectionId) {
      // Find matching anchor item and update current section
      const aItems = this.getModel('view').getProperty('/anchorBarItems');
      const sCurrentSection = null;
            
      aItems.forEach((oItem, iIndex) => {
        if (sSectionId && sSectionId.indexOf(oItem.key) !== -1) {
          sCurrentSection = oItem.key;
          oItem.current = true;
        } else {
          oItem.current = false;
        }
      });
            
      this.getModel('view').setProperty('/currentSection', sCurrentSection);
      this.getModel('view').setProperty('/anchorBarItems', aItems);
    },
        
    onAnchorBarPress: function (oEvent) {
      const oItem = oEvent.getParameter('item') || oEvent.getSource();
      const sKey = oItem.getKey ? oItem.getKey() : oItem.getCustomData()[0].getValue();
            
      this._navigateToSection(sKey);
    },
        
    _navigateToSection: function (sSectionKey) {
      const oObjectPageLayout = this.byId('objectPageLayout');
      if (!oObjectPageLayout) {
        return;
      }
            
      // Find the section to navigate to
      const aSections = oObjectPageLayout.getSections();
      const oTargetSection = null;
            
      aSections.forEach((oSection) => {
        if (oSection.getId().indexOf(sSectionKey) !== -1) {
          oTargetSection = oSection;
        }
      });
            
      if (oTargetSection) {
        // Smooth scroll to section
        if (this._anchorBarConfig.smoothScrollBehavior) {
          this._smoothScrollToSection(oTargetSection);
        } else {
          oObjectPageLayout.scrollToSection(oTargetSection.getId());
        }
                
        // Update current section and apply highlighting
        this._updateCurrentSection(oTargetSection.getId());
        this._highlightSection(oTargetSection.getId(), true);
      }
    },
        
    _smoothScrollToSection: function (oSection) {
      const oObjectPageLayout = this.byId('objectPageLayout');
      const $section = oSection.$();
            
      if ($section.length > 0) {
        const iTargetOffset = $section.offset().top - this._anchorBarConfig.scrollOffset;
        const $scrollContainer = oObjectPageLayout.$().find('.sapUxAPObjectPageScrollContainer');
                
        if ($scrollContainer.length > 0) {
          $scrollContainer.animate({
            scrollTop: iTargetOffset
          }, 500, 'easeInOutQuad');
        }
      }
    },
        
    _navigateToSectionByIndex: function (iIndex) {
      const aItems = this.getModel('view').getProperty('/anchorBarItems');
      if (aItems && aItems[iIndex] && aItems[iIndex].visible) {
        this._navigateToSection(aItems[iIndex].key);
      }
    },
        
    _navigateToPreviousSection: function () {
      const sCurrentSection = this.getModel('view').getProperty('/currentSection');
      const aItems = this.getModel('view').getProperty('/anchorBarItems');
            
      if (!sCurrentSection || !aItems) {
        return;
      }
            
      const iCurrentIndex = -1;
      for (let i = 0; i < aItems.length; i++) {
        if (aItems[i].key === sCurrentSection) {
          iCurrentIndex = i;
          break;
        }
      }
            
      if (iCurrentIndex > 0) {
        this._navigateToSection(aItems[iCurrentIndex - 1].key);
      }
    },
        
    _navigateToNextSection: function () {
      const sCurrentSection = this.getModel('view').getProperty('/currentSection');
      const aItems = this.getModel('view').getProperty('/anchorBarItems');
            
      if (!sCurrentSection || !aItems) {
        return;
      }
            
      const iCurrentIndex = -1;
      for (let i = 0; i < aItems.length; i++) {
        if (aItems[i].key === sCurrentSection) {
          iCurrentIndex = i;
          break;
        }
      }
            
      if (iCurrentIndex >= 0 && iCurrentIndex < aItems.length - 1) {
        this._navigateToSection(aItems[iCurrentIndex + 1].key);
      }
    },
        
    _focusAnchorBar: function () {
      const oObjectPageLayout = this.byId('objectPageLayout');
      if (oObjectPageLayout) {
        const oAnchorBar = oObjectPageLayout.getAnchorBar();
        if (oAnchorBar) {
          oAnchorBar.focus();
        }
      }
    },
        
    onToggleAnchorBar: function () {
      const bVisible = this.getModel('view').getProperty('/anchorBarVisible');
      this.getModel('view').setProperty('/anchorBarVisible', !bVisible);
            
      const oObjectPageLayout = this.byId('objectPageLayout');
      if (oObjectPageLayout) {
        oObjectPageLayout.setShowAnchorBar(!bVisible);
      }
    },
        
    _setupAdvancedScrolling: function () {
      const oObjectPageLayout = this.byId('objectPageLayout');
      if (!oObjectPageLayout) {
        return;
      }
            
      // Setup scroll spy for section tracking
      if (this._anchorBarConfig.enableScrollSpy) {
        this._setupScrollSpy(oObjectPageLayout);
      }
            
      // Setup scroll memory for position tracking
      if (this._anchorBarConfig.enableScrollMemory) {
        this._setupScrollMemory(oObjectPageLayout);
      }
            
      // Setup scroll direction detection
      this._setupScrollDirectionTracking(oObjectPageLayout);
    },
        
    _setupScrollSpy: function (oObjectPageLayout) {
      const that = this;
            
      // Debounced scroll handler
      this._debouncedScrollHandler = this._debounce(() => {
        that._handleScrollSpyUpdate();
      }, this._anchorBarConfig.scrollDebounce);
            
      // Attach scroll event listener
      oObjectPageLayout.attachEvent('_scroll', this._debouncedScrollHandler);
    },
        
    _setupScrollMemory: function (oObjectPageLayout) {
      // Store scroll positions for each section
      this._sectionScrollPositions = {};
            
      // Save scroll position when navigating away
      const that = this;
      oObjectPageLayout.attachNavigate((oEvent) => {
        const sFromSectionId = oEvent.getParameter('from');
        if (sFromSectionId) {
          that._saveScrollPosition(sFromSectionId);
        }
      });
    },
        
    _setupScrollDirectionTracking: function (oObjectPageLayout) {
      const that = this;
            
      // Track scroll direction for enhanced UX
      this._scrollDirectionHandler = function (oEvent) {
        const iCurrentScrollTop = that._getScrollTop();
                
        if (iCurrentScrollTop > that._lastScrollTop) {
          that._scrollDirection = 'down';
        } else if (iCurrentScrollTop < that._lastScrollTop) {
          that._scrollDirection = 'up';
        }
                
        that._lastScrollTop = iCurrentScrollTop;
        that._updateScrollDirection();
      };
            
      oObjectPageLayout.attachEvent('_scroll', this._scrollDirectionHandler);
    },
        
    _handleScrollSpyUpdate: function () {
      if (!this._anchorBarConfig.enableScrollSpy) {
        return;
      }
            
      const oObjectPageLayout = this.byId('objectPageLayout');
      const aSections = oObjectPageLayout.getSections();
      const iScrollTop = this._getScrollTop();
      const sCurrentSection = null;
            
      // Find the section currently in view
      for (let i = 0; i < aSections.length; i++) {
        const oSection = aSections[i];
        const $section = oSection.$();
                
        if ($section.length > 0) {
          const oSectionPos = $section.offset();
          const iSectionTop = oSectionPos.top - this._anchorBarConfig.scrollOffset;
          const iSectionBottom = iSectionTop + $section.outerHeight();
                    
          if (iScrollTop >= iSectionTop && iScrollTop < iSectionBottom) {
            sCurrentSection = oSection.getId();
            break;
          }
        }
      }
            
      if (sCurrentSection) {
        this._updateCurrentSection(sCurrentSection);
      }
    },
        
    _getScrollTop: function () {
      const oObjectPageLayout = this.byId('objectPageLayout');
      const $scrollContainer = oObjectPageLayout.$().find('.sapUxAPObjectPageScrollContainer');
            
      return $scrollContainer.length > 0 ? $scrollContainer.scrollTop() : 0;
    },
        
    _saveScrollPosition: function (sSectionId) {
      if (!this._anchorBarConfig.enableScrollMemory) {
        return;
      }
            
      const iScrollTop = this._getScrollTop();
      this._sectionScrollPositions[sSectionId] = iScrollTop;
    },
        
    _restoreScrollPosition: function (sSectionId) {
      if (!this._anchorBarConfig.enableScrollMemory) {
        return;
      }
            
      const iSavedPosition = this._sectionScrollPositions[sSectionId];
      if (typeof iSavedPosition === 'number') {
        this._scrollToPosition(iSavedPosition);
      }
    },
        
    _scrollToPosition: function (iPosition) {
      const oObjectPageLayout = this.byId('objectPageLayout');
      const $scrollContainer = oObjectPageLayout.$().find('.sapUxAPObjectPageScrollContainer');
            
      if ($scrollContainer.length > 0) {
        if (this._anchorBarConfig.smoothScrollBehavior) {
          $scrollContainer.animate({
            scrollTop: iPosition
          }, 300, 'easeInOutQuad');
        } else {
          $scrollContainer.scrollTop(iPosition);
        }
      }
    },
        
    _updateScrollDirection: function () {
      // Update view model with scroll direction for UI animations
      this.getModel('view').setProperty('/scrollDirection', this._scrollDirection);
    },
        
    _debounce: function (func, wait) {
      let timeout;
      return function executedFunction() {
        const context = this;
        const args = args;
        const later = function () {
          timeout = null;
          func.apply(context, args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
      };
    },
        
    onScrollToTop: function () {
      this._scrollToPosition(0);
    },
        
    onScrollToBottom: function () {
      const oObjectPageLayout = this.byId('objectPageLayout');
      const $scrollContainer = oObjectPageLayout.$().find('.sapUxAPObjectPageScrollContainer');
            
      if ($scrollContainer.length > 0) {
        const iMaxScroll = $scrollContainer[0].scrollHeight - $scrollContainer.height();
        this._scrollToPosition(iMaxScroll);
      }
    },
        
    onScrollByPage: function (bDown) {
      const oObjectPageLayout = this.byId('objectPageLayout');
      const $scrollContainer = oObjectPageLayout.$().find('.sapUxAPObjectPageScrollContainer');
            
      if ($scrollContainer.length > 0) {
        const iCurrentScroll = $scrollContainer.scrollTop();
        const iPageSize = $scrollContainer.height() * 0.8; // 80% of viewport
        const iNewPosition = bDown ? 
          iCurrentScroll + iPageSize : 
          iCurrentScroll - iPageSize;
                
        this._scrollToPosition(Math.max(0, iNewPosition));
      }
    },
        
    onScrollToSection: function (sSectionKey) {
      // Public method for programmatic section scrolling
      this._navigateToSection(sSectionKey);
    },
        
    getCurrentScrollInfo: function () {
      return {
        scrollTop: this._getScrollTop(),
        scrollDirection: this._scrollDirection,
        currentSection: this.getModel('view').getProperty('/currentSection'),
        scrollPositions: Object.assign({}, this._sectionScrollPositions)
      };
    },
        
    _initializeSectionHighlighting: function () {
      // Initialize section highlighting system
      this._highlightConfig = this.getModel('view').getProperty('/sectionHighlighting');
            
      // Add CSS classes for highlighting
      this._addHighlightingStyles();
            
      // Setup intersection observer for advanced highlighting
      this._setupIntersectionObserver();
            
      // Setup highlighting animations
      this._setupHighlightAnimations();
    },
        
    _addHighlightingStyles: function () {
      // Add CSS styles for section highlighting
      const sStyles = `
                .a2a-section-highlighted {
                    background-color: rgba(0, 123, 191, 0.05) !important;
                    border-left: 4px solid #007bff !important;
                    transition: all 200ms ease-in-out !important;
                    box-shadow: 0 2px 8px rgba(0, 123, 191, 0.15) !important;
                }
                
                .a2a-section-fade {
                    background-color: transparent !important;
                    border-left: none !important;
                    transition: all 200ms ease-in-out !important;
                    box-shadow: none !important;
                }
                
                .a2a-section-highlight-animation {
                    animation: sectionHighlight 500ms ease-in-out;
                }
                
                @keyframes sectionHighlight {
                    0% { 
                        background-color: transparent;
                        border-left-width: 0px;
                    }
                    50% { 
                        background-color: rgba(0, 123, 191, 0.1);
                        border-left-width: 2px;
                    }
                    100% { 
                        background-color: rgba(0, 123, 191, 0.05);
                        border-left-width: 4px;
                    }
                }
                
                .a2a-anchor-item-highlighted {
                    background-color: #007bff !important;
                    color: white !important;
                    font-weight: bold !important;
                    transform: scale(1.05) !important;
                    transition: all 150ms ease-in-out !important;
                }
                
                .a2a-section-title-highlighted {
                    color: #007bff !important;
                    font-weight: 600 !important;
                    transition: color 200ms ease-in-out !important;
                }
            `;
            
      // Inject styles into head
      if (!document.getElementById('sectionHighlightStyles')) {
        const oStyleElement = document.createElement('style');
        oStyleElement.id = 'sectionHighlightStyles';
        oStyleElement.textContent = sStyles;
        document.head.appendChild(oStyleElement);
      }
    },
        
    _setupIntersectionObserver: function () {
      // Use Intersection Observer for more accurate section detection
      if ('IntersectionObserver' in window) {
        const that = this;
                
        this._intersectionObserver = new IntersectionObserver(((entries) => {
          that._handleIntersectionChanges(entries);
        }), {
          root: null,
          rootMargin: '-60px 0px -60px 0px', // Account for header
          threshold: [0.1, 0.5, 0.9] // Multiple thresholds for better detection
        });
                
        // Observe all sections after view is rendered
        this.getView().addEventDelegate({
          onAfterRendering: function () {
            that._observeSections();
          }
        });
      }
    },
        
    _observeSections: function () {
      if (!this._intersectionObserver) {
        return;
      }
            
      const oObjectPageLayout = this.byId('objectPageLayout');
      if (oObjectPageLayout) {
        const aSections = oObjectPageLayout.getSections();
        aSections.forEach((oSection) => {
          const $section = oSection.$();
          if ($section.length > 0) {
            this._intersectionObserver.observe($section[0]);
          }
        });
      }
    },
        
    _handleIntersectionChanges: function (entries) {
      const oMostVisibleSection = null;
      const fMaxVisibility = 0;
            
      entries.forEach((entry) => {
        if (entry.isIntersecting && entry.intersectionRatio > fMaxVisibility) {
          fMaxVisibility = entry.intersectionRatio;
          oMostVisibleSection = entry.target;
        }
      });
            
      if (oMostVisibleSection) {
        const sSectionId = oMostVisibleSection.id;
        this._highlightSection(sSectionId);
        this._updateCurrentSection(sSectionId);
      }
    },
        
    _setupHighlightAnimations: function () {
      // Setup animation system for section highlighting
      this._highlightTransitions = {
        fadeIn: 'a2a-section-highlighted',
        fadeOut: 'a2a-section-fade',
        pulse: 'a2a-section-highlight-animation'
      };
    },
        
    _highlightSection: function (sSectionId, bAnimate = true) {
      if (!this._highlightConfig.enabled) {
        return;
      }
            
      const oObjectPageLayout = this.byId('objectPageLayout');
      if (!oObjectPageLayout) {
        return;
      }
            
      // Remove previous highlighting
      this._removeAllSectionHighlights();
            
      // Find and highlight current section
      const aSections = oObjectPageLayout.getSections();
      const oTargetSection = null;
            
      aSections.forEach((oSection) => {
        if (oSection.getId() === sSectionId) {
          oTargetSection = oSection;
        }
      });
            
      if (oTargetSection) {
        this._applySectionHighlight(oTargetSection, bAnimate);
        this._highlightSectionTitle(oTargetSection);
        this._updateHighlightedSection(sSectionId);
      }
    },
        
    _applySectionHighlight: function (oSection, bAnimate) {
      const $section = oSection.$();
      if ($section.length > 0) {
        if (bAnimate) {
          // Apply animation class first
          $section.addClass(this._highlightTransitions.pulse);
                    
          // Remove animation class after completion
          setTimeout(() => {
            $section.removeClass(this._highlightTransitions.pulse);
          }, 500);
        }
                
        // Apply highlight class
        $section.addClass(this._highlightConfig.highlightClass);
      }
    },
        
    _highlightSectionTitle: function (oSection) {
      const $sectionTitle = oSection.$().find('.sapUxAPObjectPageSectionTitle');
      if ($sectionTitle.length > 0) {
        $sectionTitle.addClass('a2a-section-title-highlighted');
      }
    },
        
    _removeAllSectionHighlights: function () {
      const oObjectPageLayout = this.byId('objectPageLayout');
      if (!oObjectPageLayout) {
        return;
      }
            
      const aSections = oObjectPageLayout.getSections();
      aSections.forEach((oSection) => {
        const $section = oSection.$();
        if ($section.length > 0) {
          // Remove highlight classes
          $section.removeClass(this._highlightConfig.highlightClass);
          $section.removeClass(this._highlightConfig.fadeClass);
          $section.removeClass(this._highlightTransitions.pulse);
                    
          // Remove title highlighting
          const $sectionTitle = $section.find('.sapUxAPObjectPageSectionTitle');
          $sectionTitle.removeClass('a2a-section-title-highlighted');
        }
      });
    },
        
    _updateHighlightedSection: function (sSectionId) {
      this.getModel('view').setProperty('/sectionHighlighting/currentHighlighted', sSectionId);
    },
        
    onToggleSectionHighlighting: function () {
      const bEnabled = this.getModel('view').getProperty('/sectionHighlighting/enabled');
      this.getModel('view').setProperty('/sectionHighlighting/enabled', !bEnabled);
            
      if (!bEnabled) {
        // Re-highlight current section
        const sCurrentSection = this.getModel('view').getProperty('/currentSection');
        if (sCurrentSection) {
          this._highlightSection(sCurrentSection, false);
        }
      } else {
        // Remove all highlights
        this._removeAllSectionHighlights();
      }
    },
        
    onHighlightSection: function (oEvent) {
      // Manual section highlighting trigger
      const sSectionKey = oEvent.getParameter('sectionKey') || oEvent.getSource().data('sectionKey');
      if (sSectionKey) {
        const oObjectPageLayout = this.byId('objectPageLayout');
        const aSections = oObjectPageLayout.getSections();
                
        aSections.forEach((oSection) => {
          if (oSection.getId().indexOf(sSectionKey) !== -1) {
            this._highlightSection(oSection.getId(), true);
          }
        });
      }
    },
        
    _enhanceAnchorBarHighlighting: function () {
      // Enhanced anchor bar highlighting coordination
      const oObjectPageLayout = this.byId('objectPageLayout');
      if (!oObjectPageLayout) {
        return;
      }
            
      // Override default anchor bar highlighting
      const oAnchorBar = oObjectPageLayout.getAnchorBar();
      if (oAnchorBar) {
        const aButtons = oAnchorBar.getContent();
        aButtons.forEach((oButton, iIndex) => {
          oButton.attachPress(() => {
            this._highlightAnchorButton(iIndex);
          });
        });
      }
    },
        
    _highlightAnchorButton: function (iIndex) {
      const oObjectPageLayout = this.byId('objectPageLayout');
      const oAnchorBar = oObjectPageLayout.getAnchorBar();
            
      if (oAnchorBar) {
        const aButtons = oAnchorBar.getContent();
                
        // Remove previous highlighting
        aButtons.forEach((oButton) => {
          oButton.removeStyleClass('a2a-anchor-item-highlighted');
        });
                
        // Add highlighting to selected button
        if (aButtons[iIndex]) {
          aButtons[iIndex].addStyleClass('a2a-anchor-item-highlighted');
        }
      }
    },
        
    getHighlightingInfo: function () {
      return {
        enabled: this.getModel('view').getProperty('/sectionHighlighting/enabled'),
        currentHighlighted: this.getModel('view').getProperty('/sectionHighlighting/currentHighlighted'),
        animationDuration: this.getModel('view').getProperty('/sectionHighlighting/animationDuration')
      };
    },
        
    _initializeSectionCollapse: function () {
      // Initialize section expand/collapse system
      this._collapseConfig = this.getModel('view').getProperty('/sectionCollapse');
            
      // Add CSS styles for collapse animations
      this._addCollapseStyles();
            
      // Setup collapse controls after view rendering
      this.getView().addEventDelegate({
        onAfterRendering: function () {
          this._setupSectionCollapseControls();
          this._restoreCollapsedState();
        }.bind(this)
      });
            
      // Setup keyboard shortcuts for expand/collapse
      this._setupCollapseKeyboardShortcuts();
    },
        
    _addCollapseStyles: function () {
      // Add CSS styles for section collapse animations
      const sStyles = `
                .a2a-section-collapsed .sapUxAPObjectPageSectionContent {
                    max-height: 0 !important;
                    overflow: hidden !important;
                    opacity: 0 !important;
                    transition: all 300ms ease-in-out !important;
                }
                
                .a2a-section-expanded .sapUxAPObjectPageSectionContent {
                    max-height: none !important;
                    opacity: 1 !important;
                    transition: all 300ms ease-in-out !important;
                }
                
                .a2a-section-collapsing {
                    transition: all 300ms ease-in-out !important;
                }
                
                .a2a-collapse-toggle {
                    cursor: pointer !important;
                    transition: transform 200ms ease-in-out !important;
                }
                
                .a2a-collapse-toggle.collapsed {
                    transform: rotate(-90deg) !important;
                }
                
                .a2a-section-header-collapsed {
                    opacity: 0.7 !important;
                    transition: opacity 200ms ease-in-out !important;
                }
                
                .a2a-section-title-container {
                    display: flex !important;
                    align-items: center !important;
                    cursor: pointer !important;
                }
                
                .a2a-section-title-container:hover {
                    background-color: rgba(0, 123, 191, 0.05) !important;
                }
                
                .a2a-collapse-icon {
                    margin-right: 8px !important;
                    transition: transform 200ms ease-in-out !important;
                }
                
                .a2a-section-badge {
                    margin-left: auto !important;
                    font-size: 0.75rem !important;
                    background-color: #007bff !important;
                    color: white !important;
                    padding: 2px 8px !important;
                    border-radius: 12px !important;
                }
            `;
            
      // Inject styles into head
      if (!document.getElementById('sectionCollapseStyles')) {
        const oStyleElement = document.createElement('style');
        oStyleElement.id = 'sectionCollapseStyles';
        oStyleElement.textContent = sStyles;
        document.head.appendChild(oStyleElement);
      }
    },
        
    _setupSectionCollapseControls: function () {
      const oObjectPageLayout = this.byId('objectPageLayout');
      if (!oObjectPageLayout) {
        return;
      }
            
      const aSections = oObjectPageLayout.getSections();
      aSections.forEach((oSection) => {
        this._addCollapseControlToSection(oSection);
      });
    },
        
    _addCollapseControlToSection: function (oSection) {
      const $section = oSection.$();
      if ($section.length === 0) {
        return;
      }
            
      const $header = $section.find('.sapUxAPObjectPageSectionHeader');
      const $title = $header.find('.sapUxAPObjectPageSectionTitle');
            
      if ($title.length > 0 && !$title.hasClass('a2a-collapse-enhanced')) {
        // Create collapse toggle container
        const $titleContainer = $('<div class="a2a-section-title-container"></div>');
        const $collapseIcon = $('<span class="a2a-collapse-icon sap-icon sap-icon--navigation-down-arrow"></span>');
                
        // Wrap title and add icon
        $title.wrap($titleContainer);
        $title.parent().prepend($collapseIcon);
                
        // Add click handler
        const that = this;
        $title.parent().on('click', (e) => {
          e.preventDefault();
          e.stopPropagation();
          that._toggleSectionCollapse(oSection.getId());
        });
                
        // Mark as enhanced
        $title.addClass('a2a-collapse-enhanced');
                
        // Add section badge if applicable
        this._addSectionBadge(oSection);
      }
    },
        
    _addSectionBadge: function (oSection) {
      const sSectionId = oSection.getId();
      const iBadgeCount = this._getSectionBadgeCount(sSectionId);
            
      if (iBadgeCount > 0) {
        const $header = oSection.$().find('.sapUxAPObjectPageSectionHeader');
        const $titleContainer = $header.find('.a2a-section-title-container');
                
        if ($titleContainer.length > 0 && !$titleContainer.find('.a2a-section-badge').length) {
          const $badge = $(`<span class="a2a-section-badge">${  iBadgeCount  }</span>`);
          $titleContainer.append($badge);
        }
      }
    },
        
    _getSectionBadgeCount: function (sSectionId) {
      const oProjectData = this.getModel('project').getData();
      if (!oProjectData) {
        return 0;
      }
            
      if (sSectionId.indexOf('agents') !== -1) {
        return oProjectData.agents ? oProjectData.agents.length : 0;
      } else if (sSectionId.indexOf('workflows') !== -1) {
        return oProjectData.workflows ? oProjectData.workflows.length : 0;
      } else if (sSectionId.indexOf('team') !== -1) {
        return oProjectData.members ? oProjectData.members.length : 0;
      } else if (sSectionId.indexOf('activity') !== -1) {
        return oProjectData.activities ? oProjectData.activities.length : 0;
      }
      return 0;
    },
        
    _toggleSectionCollapse: function (sSectionId) {
      const oObjectPageLayout = this.byId('objectPageLayout');
      const oSection = oObjectPageLayout.getSections().find((s) => {
        return s.getId() === sSectionId;
      });
            
      if (!oSection) {
        return;
      }
            
      const $section = oSection.$();
      const bIsCollapsed = $section.hasClass('a2a-section-collapsed');
            
      if (bIsCollapsed) {
        this._expandSection(sSectionId);
      } else {
        this._collapseSection(sSectionId);
      }
    },
        
    _collapseSection: function (sSectionId) {
      const oObjectPageLayout = this.byId('objectPageLayout');
      const oSection = oObjectPageLayout.getSections().find((s) => {
        return s.getId() === sSectionId;
      });
            
      if (!oSection) {
        return;
      }
            
      const $section = oSection.$();
      const $content = $section.find('.sapUxAPObjectPageSectionContent');
      const $icon = $section.find('.a2a-collapse-icon');
      const $header = $section.find('.sapUxAPObjectPageSectionHeader');
            
      // Add collapsing animation
      $section.addClass('a2a-section-collapsing');
      $content.css('max-height', `${$content.height()  }px`);
            
      // Trigger collapse animation
      setTimeout(() => {
        $section.addClass('a2a-section-collapsed');
        $section.removeClass('a2a-section-expanded');
        $icon.addClass('collapsed');
        $header.addClass('a2a-section-header-collapsed');
        $content.css('max-height', '0px');
      }, 10);
            
      // Clean up animation classes
      setTimeout(() => {
        $section.removeClass('a2a-section-collapsing');
      }, this._collapseConfig.animationDuration + 50);
            
      // Update collapsed sections state
      this._updateCollapsedSections(sSectionId, true);
    },
        
    _expandSection: function (sSectionId) {
      const oObjectPageLayout = this.byId('objectPageLayout');
      const oSection = oObjectPageLayout.getSections().find((s) => {
        return s.getId() === sSectionId;
      });
            
      if (!oSection) {
        return;
      }
            
      const $section = oSection.$();
      const $content = $section.find('.sapUxAPObjectPageSectionContent');
      const $icon = $section.find('.a2a-collapse-icon');
      const $header = $section.find('.sapUxAPObjectPageSectionHeader');
            
      // Add expanding animation
      $section.addClass('a2a-section-collapsing');
            
      // Get natural height for animation
      $content.css('max-height', 'none');
      const iNaturalHeight = $content.height();
      $content.css('max-height', '0px');
            
      // Trigger expand animation
      setTimeout(() => {
        $section.removeClass('a2a-section-collapsed');
        $section.addClass('a2a-section-expanded');
        $icon.removeClass('collapsed');
        $header.removeClass('a2a-section-header-collapsed');
        $content.css('max-height', `${iNaturalHeight  }px`);
      }, 10);
            
      // Complete expansion
      setTimeout(() => {
        $content.css('max-height', 'none');
        $section.removeClass('a2a-section-collapsing');
      }, this._collapseConfig.animationDuration + 50);
            
      // Update collapsed sections state
      this._updateCollapsedSections(sSectionId, false);
    },
        
    _updateCollapsedSections: function (sSectionId, bCollapsed) {
      const aCollapsed = this.getModel('view').getProperty('/sectionCollapse/collapsedSections');
      const iIndex = aCollapsed.indexOf(sSectionId);
            
      if (bCollapsed && iIndex === -1) {
        aCollapsed.push(sSectionId);
      } else if (!bCollapsed && iIndex !== -1) {
        aCollapsed.splice(iIndex, 1);
      }
            
      this.getModel('view').setProperty('/sectionCollapse/collapsedSections', aCollapsed);
            
      // Persist state if enabled
      if (this._collapseConfig.persistState) {
        this._saveCollapsedState();
      }
    },
        
    _saveCollapsedState: function () {
      const aCollapsed = this.getModel('view').getProperty('/sectionCollapse/collapsedSections');
      const sProjectId = this._projectId || 'default';
            
      try {
        localStorage.setItem(`a2a-collapsed-sections-${  sProjectId}`, JSON.stringify(aCollapsed));
      } catch (e) {
        // Handle storage errors gracefully
      }
    },
        
    _restoreCollapsedState: function () {
      if (!this._collapseConfig.persistState) {
        return;
      }
            
      const sProjectId = this._projectId || 'default';
            
      try {
        const sStoredState = localStorage.getItem(`a2a-collapsed-sections-${  sProjectId}`);
        if (sStoredState) {
          const aCollapsed = JSON.parse(sStoredState);
          this.getModel('view').setProperty('/sectionCollapse/collapsedSections', aCollapsed);
                    
          // Apply collapsed state to sections
          aCollapsed.forEach((sSectionId) => {
            this._collapseSection(sSectionId);
          });
        }
      } catch (e) {
        // Handle parsing errors gracefully
      }
    },
        
    _setupCollapseKeyboardShortcuts: function () {
      // Add keyboard shortcuts for section collapse
      this.getView().addEventDelegate({
        onkeydown: function (oEvent) {
          // Ctrl+Shift+C - Collapse all sections
          if (oEvent.ctrlKey && oEvent.shiftKey && oEvent.which === 67) {
            this.onCollapseAllSections();
            oEvent.preventDefault();
          }
          // Ctrl+Shift+E - Expand all sections
          else if (oEvent.ctrlKey && oEvent.shiftKey && oEvent.which === 69) {
            this.onExpandAllSections();
            oEvent.preventDefault();
          }
          // Ctrl+[ - Collapse current section
          else if (oEvent.ctrlKey && oEvent.which === 219) {
            const sCurrentSection = this.getModel('view').getProperty('/currentSection');
            if (sCurrentSection) {
              this._collapseSection(sCurrentSection);
            }
            oEvent.preventDefault();
          }
          // Ctrl+] - Expand current section
          else if (oEvent.ctrlKey && oEvent.which === 221) {
            const sCurrentSection = this.getModel('view').getProperty('/currentSection');
            if (sCurrentSection) {
              this._expandSection(sCurrentSection);
            }
            oEvent.preventDefault();
          }
        }.bind(this)
      });
    },
        
    onCollapseAllSections: function () {
      const oObjectPageLayout = this.byId('objectPageLayout');
      const aSections = oObjectPageLayout.getSections();
            
      aSections.forEach((oSection, iIndex) => {
        // Add slight delay for staggered animation
        setTimeout(() => {
          this._collapseSection(oSection.getId());
        }, iIndex * 100);
      });
    },
        
    onExpandAllSections: function () {
      const oObjectPageLayout = this.byId('objectPageLayout');
      const aSections = oObjectPageLayout.getSections();
            
      aSections.forEach((oSection, iIndex) => {
        // Add slight delay for staggered animation
        setTimeout(() => {
          this._expandSection(oSection.getId());
        }, iIndex * 100);
      });
    },
        
    onToggleCollapseMode: function () {
      const bEnabled = this.getModel('view').getProperty('/sectionCollapse/enabled');
      this.getModel('view').setProperty('/sectionCollapse/enabled', !bEnabled);
            
      if (bEnabled) {
        // Expand all sections when disabling collapse mode
        this.onExpandAllSections();
      }
    },
        
    getSectionCollapseInfo: function () {
      return {
        enabled: this.getModel('view').getProperty('/sectionCollapse/enabled'),
        collapsedSections: this.getModel('view').getProperty('/sectionCollapse/collapsedSections'),
        allowMultipleExpanded: this.getModel('view').getProperty('/sectionCollapse/allowMultipleExpanded')
      };
    },
        
    _initializeLazyLoading: function () {
      // Initialize lazy loading system
      this._lazyConfig = this.getModel('view').getProperty('/lazyLoading');
            
      // Setup loading placeholders
      this._addLazyLoadingStyles();
            
      // Initialize section lazy loading after rendering
      this.getView().addEventDelegate({
        onAfterRendering: function () {
          this._setupLazyLoadedSections();
          this._setupLazyLoadingObserver();
        }.bind(this)
      });
            
      // Setup performance monitoring
      this._initializeLazyLoadingMetrics();
    },
        
    _addLazyLoadingStyles: function () {
      // Add CSS styles for lazy loading placeholders and animations
      const sStyles = `
                .a2a-section-lazy-loading {
                    min-height: 200px !important;
                    display: flex !important;
                    align-items: center !important;
                    justify-content: center !important;
                    background-color: #f8f9fa !important;
                    border: 1px dashed #dee2e6 !important;
                    border-radius: 4px !important;
                    margin: 16px !important;
                }
                
                .a2a-section-lazy-placeholder {
                    text-align: center !important;
                    color: #6c757d !important;
                }
                
                .a2a-lazy-loading-spinner {
                    display: inline-block !important;
                    width: 20px !important;
                    height: 20px !important;
                    border: 3px solid #f3f3f3 !important;
                    border-top: 3px solid #007bff !important;
                    border-radius: 50% !important;
                    animation: lazySpinner 1s linear infinite !important;
                    margin-right: 8px !important;
                }
                
                @keyframes lazySpinner {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                
                .a2a-section-lazy-loaded {
                    animation: lazyFadeIn 500ms ease-in-out !important;
                }
                
                @keyframes lazyFadeIn {
                    0% { 
                        opacity: 0;
                        transform: translateY(20px);
                    }
                    100% { 
                        opacity: 1;
                        transform: translateY(0);
                    }
                }
                
                .a2a-section-lazy-error {
                    background-color: #f8d7da !important;
                    border-color: #f5c6cb !important;
                    color: #721c24 !important;
                }
                
                .a2a-lazy-retry-button {
                    margin-top: 12px !important;
                    background-color: #007bff !important;
                    color: white !important;
                    border: none !important;
                    padding: 8px 16px !important;
                    border-radius: 4px !important;
                    cursor: pointer !important;
                }
                
                .a2a-lazy-retry-button:hover {
                    background-color: #0056b3 !important;
                }
                
                .a2a-section-skeleton {
                    background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%) !important;
                    background-size: 200% 100% !important;
                    animation: skeletonLoading 1.5s infinite !important;
                }
                
                @keyframes skeletonLoading {
                    0% { background-position: 200% 0; }
                    100% { background-position: -200% 0; }
                }
            `;
            
      // Inject styles into head
      if (!document.getElementById('lazyLoadingStyles')) {
        const oStyleElement = document.createElement('style');
        oStyleElement.id = 'lazyLoadingStyles';
        oStyleElement.textContent = sStyles;
        document.head.appendChild(oStyleElement);
      }
    },
        
    _setupLazyLoadedSections: function () {
      const oObjectPageLayout = this.byId('objectPageLayout');
      if (!oObjectPageLayout) {
        return;
      }
            
      const aSections = oObjectPageLayout.getSections();
      aSections.forEach((oSection, iIndex) => {
        // First section and visible sections load immediately
        if (iIndex === 0 || this._isSectionVisible(oSection)) {
          this._loadSectionContent(oSection.getId(), false);
        } else {
          this._createLazyPlaceholder(oSection);
        }
      });
    },
        
    _isSectionVisible: function (oSection) {
      const $section = oSection.$();
      if ($section.length === 0) {
        return false;
      }
            
      const oRect = $section[0].getBoundingClientRect();
      const iViewportHeight = window.innerHeight || document.documentElement.clientHeight;
            
      return (oRect.top < iViewportHeight && oRect.bottom > 0);
    },
        
    _createLazyPlaceholder: function (oSection) {
      const sSectionId = oSection.getId();
      const $section = oSection.$();
      const $content = $section.find('.sapUxAPObjectPageSectionContent');
            
      if ($content.length > 0) {
        // Store original content for later loading
        this._storeSectionContent(sSectionId, $content.html());
                
        // Create placeholder content
        const sPlaceholderHtml = this._createPlaceholderHtml(sSectionId);
        $content.html(sPlaceholderHtml);
        $content.addClass('a2a-section-lazy-loading');
      }
    },
        
    _createPlaceholderHtml: function (sSectionId) {
      const sPlaceholderType = this._getPlaceholderType(sSectionId);
      const sTitle = this._getSectionDisplayName(sSectionId);
            
      return `
                <div class="a2a-section-lazy-placeholder">
                    <div class="a2a-lazy-loading-spinner"></div>
                    <div>
                        <h4>Loading ${sTitle}...</h4>
                        <p>Content will appear shortly</p>
                        <div class="a2a-section-skeleton" style="height: 100px; margin: 16px 0; border-radius: 4px;"></div>
                        <div class="a2a-section-skeleton" style="height: 60px; margin: 8px 0; border-radius: 4px;"></div>
                    </div>
                </div>
            `;
    },
        
    _getPlaceholderType: function (sSectionId) {
      if (sSectionId.indexOf('agents') !== -1) {
        return 'table';
      }
      if (sSectionId.indexOf('workflows') !== -1) {
        return 'grid';
      }
      if (sSectionId.indexOf('team') !== -1) {
        return 'list';
      }
      if (sSectionId.indexOf('metrics') !== -1) {
        return 'charts';
      }
      if (sSectionId.indexOf('activity') !== -1) {
        return 'timeline';
      }
      return 'content';
    },
        
    _getSectionDisplayName: function (sSectionId) {
      if (sSectionId.indexOf('general') !== -1) {
        return 'General Information';
      }
      if (sSectionId.indexOf('agents') !== -1) {
        return 'Agents';
      }
      if (sSectionId.indexOf('workflows') !== -1) {
        return 'Workflows';
      }
      if (sSectionId.indexOf('team') !== -1) {
        return 'Team Members';
      }
      if (sSectionId.indexOf('metrics') !== -1) {
        return 'Metrics & Analytics';
      }
      if (sSectionId.indexOf('activity') !== -1) {
        return 'Recent Activity';
      }
      return 'Section Content';
    },
        
    _storeSectionContent: function (sSectionId, sContent) {
      if (!this._sectionContentCache) {
        this._sectionContentCache = {};
      }
      this._sectionContentCache[sSectionId] = sContent;
    },
        
    _setupLazyLoadingObserver: function () {
      if ('IntersectionObserver' in window && this._lazyConfig.enabled) {
        const that = this;
                
        this._lazyObserver = new IntersectionObserver(((entries) => {
          entries.forEach((entry) => {
            if (entry.isIntersecting) {
              const sSectionId = entry.target.id;
              that._loadSectionContent(sSectionId, true);
            }
          });
        }), {
          root: null,
          rootMargin: `${this._lazyConfig.preloadDistance  }px`,
          threshold: this._lazyConfig.threshold
        });
                
        // Observe all sections
        const oObjectPageLayout = this.byId('objectPageLayout');
        const aSections = oObjectPageLayout.getSections();
        aSections.forEach((oSection) => {
          const $section = oSection.$();
          if ($section.length > 0) {
            this._lazyObserver.observe($section[0]);
          }
        });
      }
    },
        
    _loadSectionContent: function (sSectionId, bAnimate) {
      // Check if already loaded
      const aLoaded = this.getModel('view').getProperty('/lazyLoading/loadedSections');
      if (aLoaded.indexOf(sSectionId) !== -1) {
        return Promise.resolve();
      }
            
      // Mark as loading
      this._setSectionLoadingState(sSectionId, 'loading');
            
      // Simulate API call or perform actual data loading
      return this._fetchSectionData(sSectionId)
        .then((oData) => {
          this._renderSectionContent(sSectionId, oData, bAnimate);
          this._setSectionLoadingState(sSectionId, 'loaded');
                    
          // Add to loaded sections
          aLoaded.push(sSectionId);
          this.getModel('view').setProperty('/lazyLoading/loadedSections', aLoaded);
                    
          // Track performance metrics
          this._trackLazyLoadingMetrics(sSectionId, 'success');
        })
        .catch((oError) => {
          this._renderSectionError(sSectionId, oError);
          this._setSectionLoadingState(sSectionId, 'error');
          this._trackLazyLoadingMetrics(sSectionId, 'error');
        });
    },
        
    _fetchSectionData: function (sSectionId) {
      // Simulate different loading times for different sections
      const iDelay = this._getSectionLoadTime(sSectionId);
            
      return new Promise((resolve, reject) => {
        setTimeout(() => {
          // Simulate occasional failures for testing
          if (Math.random() < 0.1) { // 10% failure rate
            reject(new Error('Failed to load section data'));
            return;
          }
                    
          // Return section-specific data
          const oData = this._generateSectionData(sSectionId);
          resolve(oData);
        }, iDelay);
      });
    },
        
    _getSectionLoadTime: function (sSectionId) {
      // Different sections have different complexity/load times
      if (sSectionId.indexOf('metrics') !== -1) {
        return 1500;
      } // Metrics take longer
      if (sSectionId.indexOf('activity') !== -1) {
        return 1200;
      } // Activity timeline
      if (sSectionId.indexOf('agents') !== -1) {
        return 800;
      }    // Agent table
      if (sSectionId.indexOf('workflows') !== -1) {
        return 600;
      } // Workflow grid
      if (sSectionId.indexOf('team') !== -1) {
        return 400;
      }      // Team list
      return 300; // General content
    },
        
    _generateSectionData: function (sSectionId) {
      // Generate section-specific data
      const oProjectData = this.getModel('project').getData();
      return {
        sectionId: sSectionId,
        timestamp: new Date(),
        data: oProjectData,
        loadTime: Date.now()
      };
    },
        
    _renderSectionContent: function (sSectionId, oData, bAnimate) {
      const oObjectPageLayout = this.byId('objectPageLayout');
      const oSection = oObjectPageLayout.getSections().find((s) => {
        return s.getId() === sSectionId;
      });
            
      if (!oSection) {
        return;
      }
            
      const $section = oSection.$();
      const $content = $section.find('.sapUxAPObjectPageSectionContent');
            
      if ($content.length > 0) {
        // Get original content or generate new content
        const sOriginalContent = this._sectionContentCache[sSectionId] || this._generateSectionHTML(sSectionId, oData);
                
        // Remove placeholder classes
        $content.removeClass('a2a-section-lazy-loading a2a-section-lazy-error');
                
        if (bAnimate) {
          // Fade out placeholder, then fade in content
          $content.fadeOut(200, () => {
            $content.html(sOriginalContent);
            $content.addClass('a2a-section-lazy-loaded');
            $content.fadeIn(300);
          });
        } else {
          $content.html(sOriginalContent);
        }
      }
    },
        
    _generateSectionHTML: function (sSectionId, oData) {
      // Generate HTML content based on section type
      if (sSectionId.indexOf('metrics') !== -1) {
        return this._generateMetricsHTML(oData);
      } else if (sSectionId.indexOf('activity') !== -1) {
        return this._generateActivityHTML(oData);
      }
      // Return default content for other sections
      return `<p>Content loaded successfully at ${  new Date().toLocaleTimeString()  }</p>`;
    },
        
    _generateMetricsHTML: function (oData) {
      return `
                <div class="sapUiMediumMargin">
                    <h4>Performance Metrics</h4>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-top: 16px;">
                        <div style="padding: 16px; border: 1px solid #ddd; border-radius: 4px;">
                            <h5>Success Rate</h5>
                            <div style="font-size: 2em; color: #28a745;">${Math.round(Math.random() * 20 + 80)}%</div>
                        </div>
                        <div style="padding: 16px; border: 1px solid #ddd; border-radius: 4px;">
                            <h5>Response Time</h5>
                            <div style="font-size: 2em; color: #007bff;">${Math.round(Math.random() * 200 + 100)}ms</div>
                        </div>
                        <div style="padding: 16px; border: 1px solid #ddd; border-radius: 4px;">
                            <h5>Total Requests</h5>
                            <div style="font-size: 2em; color: #6c757d;">${Math.round(Math.random() * 10000 + 5000)}</div>
                        </div>
                    </div>
                </div>
            `;
    },
        
    _generateActivityHTML: function (oData) {
      const aActivities = [];
      for (let i = 0; i < 5; i++) {
        aActivities.push(`
                    <div style="display: flex; padding: 12px; border-bottom: 1px solid #eee;">
                        <div style="width: 40px; height: 40px; border-radius: 50%; background: #007bff; margin-right: 12px;"></div>
                        <div>
                            <h6>Activity ${i + 1}</h6>
                            <p style="margin: 0; color: #666;">Generated at ${new Date().toLocaleTimeString()}</p>
                        </div>
                    </div>
                `);
      }
      return `<div class="sapUiMediumMargin"><h4>Recent Activity</h4>${  aActivities.join('')  }</div>`;
    },
        
    _renderSectionError: function (sSectionId, oError) {
      const oSection = this.byId('objectPageLayout').getSections().find((s) => {
        return s.getId() === sSectionId;
      });
            
      if (!oSection) {
        return;
      }
            
      const $content = oSection.$().find('.sapUxAPObjectPageSectionContent');
      const sErrorHtml = `
                <div class="a2a-section-lazy-placeholder a2a-section-lazy-error">
                    <div>
                        <h4>Failed to Load Content</h4>
                        <p>${oError.message}</p>
                        <button class="a2a-lazy-retry-button" onclick="window.a2aRetrySection('${sSectionId}')">
                            Retry Loading
                        </button>
                    </div>
                </div>
            `;
            
      $content.removeClass('a2a-section-lazy-loading');
      $content.addClass('a2a-section-lazy-error');
      $content.html(sErrorHtml);
            
      // Setup retry function
      window.a2aRetrySection = function (sSectionId) {
        this._retrySectionLoad(sSectionId);
      }.bind(this);
    },
        
    _retrySectionLoad: function (sSectionId) {
      // Remove from loaded sections to allow retry
      const aLoaded = this.getModel('view').getProperty('/lazyLoading/loadedSections');
      const iIndex = aLoaded.indexOf(sSectionId);
      if (iIndex !== -1) {
        aLoaded.splice(iIndex, 1);
        this.getModel('view').setProperty('/lazyLoading/loadedSections', aLoaded);
      }
            
      // Recreate placeholder and retry loading
      const oSection = this.byId('objectPageLayout').getSections().find((s) => {
        return s.getId() === sSectionId;
      });
            
      if (oSection) {
        this._createLazyPlaceholder(oSection);
        this._loadSectionContent(sSectionId, true);
      }
    },
        
    _setSectionLoadingState: function (sSectionId, sState) {
      // Update loading state in view model
      const oLoadingStates = this.getModel('view').getProperty('/lazyLoading/loadingStates') || {};
      oLoadingStates[sSectionId] = sState;
      this.getModel('view').setProperty('/lazyLoading/loadingStates', oLoadingStates);
    },
        
    _initializeLazyLoadingMetrics: function () {
      this._lazyMetrics = {
        loadTimes: {},
        errorCount: 0,
        successCount: 0,
        totalSections: 0
      };
    },
        
    _trackLazyLoadingMetrics: function (sSectionId, sResult) {
      if (sResult === 'success') {
        this._lazyMetrics.successCount++;
      } else if (sResult === 'error') {
        this._lazyMetrics.errorCount++;
      }
    },
        
    onToggleLazyLoading: function () {
      const bEnabled = this.getModel('view').getProperty('/lazyLoading/enabled');
      this.getModel('view').setProperty('/lazyLoading/enabled', !bEnabled);
            
      if (bEnabled) {
        // Load all sections immediately when disabling lazy loading
        this._loadAllSectionsImmediately();
      }
    },
        
    _loadAllSectionsImmediately: function () {
      const oObjectPageLayout = this.byId('objectPageLayout');
      const aSections = oObjectPageLayout.getSections();
            
      aSections.forEach((oSection) => {
        this._loadSectionContent(oSection.getId(), false);
      });
    },
        
    getLazyLoadingInfo: function () {
      return {
        enabled: this.getModel('view').getProperty('/lazyLoading/enabled'),
        loadedSections: this.getModel('view').getProperty('/lazyLoading/loadedSections'),
        metrics: Object.assign({}, this._lazyMetrics)
      };
    },
        
    _updateAnchorBarBadges: function () {
      // Update badges with actual counts after data is loaded
      const oProjectData = this.getModel('project').getData();
      const aItems = this.getModel('view').getProperty('/anchorBarItems');
            
      if (oProjectData && aItems) {
        aItems.forEach((oItem) => {
          switch (oItem.key) {
          case 'agents':
            oItem.badge = oProjectData.agents ? oProjectData.agents.length : 0;
            break;
          case 'workflows':
            oItem.badge = oProjectData.workflows ? oProjectData.workflows.length : 0;
            break;
          case 'team':
            oItem.badge = oProjectData.members ? oProjectData.members.length : 0;
            break;
          case 'activity':
            oItem.badge = oProjectData.activities ? oProjectData.activities.length : 0;
            break;
          }
        });
                
        this.getModel('view').setProperty('/anchorBarItems', aItems);
      }
    },

    // ===========================
    // Suggestions System Methods
    // ===========================
        
    _initializeSuggestions: function () {
      // Initialize AI-powered suggestions system
      this._suggestionsEngine = {
        fieldContextMap: {},
        suggestionHistory: [],
        learningModel: null,
        activeTimeout: null
      };
            
      // Pre-populate knowledge base with field patterns
      this._initializeSuggestionPatterns();
            
      // Setup event listeners for field interactions
      this._setupSuggestionTriggers();
    },
        
    _initializeSuggestionPatterns: function () {
      this._suggestionPatterns = {
        'name': {
          patterns: ['Project', 'Initiative', 'Program', 'System'],
          validationTips: [
            'Project names should be descriptive and unique',
            'Consider including department or business area',
            'Avoid special characters in project names'
          ],
          completionSuggestions: [
            {
              value: 'AI Assistant Development Project',
              action: 'replace',
              previewable: true
            },
            {
              value: 'Customer Portal Enhancement',
              action: 'replace', 
              previewable: true
            },
            {
              value: 'Data Analytics Platform',
              action: 'replace',
              previewable: true
            }
          ],
          advancedSuggestions: [
            {
              value: ' - Phase 1',
              action: 'append',
              trigger: 'project',
              description: 'Add phase suffix to project name'
            },
            {
              value: 'Project: ',
              action: 'insert',
              insertPosition: 0,
              description: 'Add project prefix'
            }
          ]
        },
        'description': {
          patterns: ['develop', 'enhance', 'implement', 'integrate'],
          validationTips: [
            'Descriptions should clearly explain project goals',
            'Include key stakeholders and business impact',
            'Specify success criteria where possible'
          ],
          completionSuggestions: [
            'This project aims to develop an AI-powered assistant system...',
            'Enhancement of the existing customer portal to improve...',
            'Implementation of advanced analytics capabilities for...'
          ]
        },
        'costCenter': {
          patterns: ['CC-IT-', 'CC-FIN-', 'CC-HR-', 'CC-OPS-'],
          validationTips: [
            'Use format CC-DEPT-### for consistency',
            'Ensure cost center exists in finance system',
            'Contact finance team for new cost centers'
          ],
          completionSuggestions: [
            'CC-IT-001',
            'CC-IT-002', 
            'CC-FIN-001'
          ]
        },
        'budget': {
          validationTips: [
            'Budget should align with project scope',
            'Include contingency buffer (typically 10-15%)',
            'Consider ongoing operational costs'
          ],
          optimizationSuggestions: [
            'Based on similar projects, consider increasing budget by 15% for contingency',
            'Resource costs may vary based on project timeline',
            'Consider phased delivery to distribute budget across quarters'
          ]
        }
      };
    },
        
    _setupSuggestionTriggers: function () {
      // Setup field focus handlers to trigger suggestions
      const aFieldIds = ['projectNameInput', 'projectDescInput', 'costCenterInput', 'budgetInput'];
            
      aFieldIds.forEach((sFieldId) => {
        const oControl = this.byId(sFieldId);
        if (oControl) {
          oControl.attachLiveChange(this._onFieldInputChange.bind(this));
          oControl.attachSuggest && oControl.attachSuggest(this._onSuggestionRequest.bind(this));
        }
      });
    },
        
    _onFieldInputChange: function (oEvent) {
      const oControl = oEvent.getSource();
      const sValue = oEvent.getParameter('value');
      const sFieldName = this._getFieldNameFromControl(oControl);
            
      if (!sFieldName) {
        return;
      }
            
      // Clear existing suggestion timeout
      if (this._suggestionsEngine.activeTimeout) {
        clearTimeout(this._suggestionsEngine.activeTimeout);
      }
            
      // Trigger suggestions after delay
      this._suggestionsEngine.activeTimeout = setTimeout(() => {
        this._generateContextualSuggestions(sFieldName, sValue);
      }, this.getModel('view').getProperty('/suggestions/showDelay'));
    },
        
    _generateContextualSuggestions: function (sFieldName, sCurrentValue) {
      if (!this.getModel('view').getProperty('/suggestions/enabled')) {
        return;
      }
      if (!this.getModel('view').getProperty('/editMode')) {
        return;
      }
            
      const oProjectData = this.getModel('project').getData();
      const aSuggestions = [];
            
      // Generate different types of suggestions
      aSuggestions = aSuggestions.concat(this._generateCompletionSuggestions(sFieldName, sCurrentValue));
      aSuggestions = aSuggestions.concat(this._generateValidationSuggestions(sFieldName, sCurrentValue, oProjectData));
      aSuggestions = aSuggestions.concat(this._generateOptimizationSuggestions(sFieldName, sCurrentValue, oProjectData));
            
      // Filter out rejected and suppressed suggestions
      aSuggestions = this._filterRejectedSuggestions(aSuggestions, sFieldName);
      aSuggestions = this._filterSuppressedSuggestions(aSuggestions, sFieldName);
            
      // Prioritize and limit suggestions
      aSuggestions = this._prioritizeSuggestions(aSuggestions);
      const iMaxSuggestions = this.getModel('view').getProperty('/suggestions/maxSuggestions');
      aSuggestions = aSuggestions.slice(0, iMaxSuggestions);
            
      // Store active field for rejection tracking
      this.getModel('view').setProperty('/suggestions/activeSuggestion', sFieldName);
            
      // Update suggestions model
      this.getModel('view').setProperty('/suggestions/currentSuggestions', aSuggestions);
            
      // Show suggestions UI if there are any
      if (aSuggestions.length > 0) {
        this._showSuggestionsPopover(sFieldName);
      }
    },
        
    _filterRejectedSuggestions: function (aSuggestions, sFieldName) {
      const oRejectionPatterns = this.getModel('view').getProperty('/suggestions/rejectionPatterns') || {};
            
      return aSuggestions.filter((oSuggestion) => {
        const sPatternKey = `${oSuggestion.type  }_${  oSuggestion.action}`;
        const aRejectedPatterns = oRejectionPatterns[sPatternKey] || [];
                
        // Check if this specific suggestion was rejected
        const bRejected = aRejectedPatterns.some((oRejected) => {
          return oRejected.value === oSuggestion.value && 
                           oRejected.fieldName === sFieldName &&
                           oRejected.reason !== 'wrong_timing'; // Allow timing-related rejections to retry later
        });
                
        return !bRejected;
      });
    },
        
    _filterSuppressedSuggestions: function (aSuggestions, sFieldName) {
      const oSuppressionData = this.getModel('view').getProperty('/suggestions/temporarySuppression') || {};
      const iCurrentTime = Date.now();
            
      return aSuggestions.filter((oSuggestion) => {
        const sKey = `${sFieldName  }_${  oSuggestion.type}`;
        const oSuppression = oSuppressionData[sKey];
                
        if (!oSuppression) {
          return true;
        }
                
        // Check if suppression period has expired
        return iCurrentTime > oSuppression.suppressedUntil;
      });
    },
        
    _generateCompletionSuggestions: function (sFieldName, sCurrentValue) {
      const aSuggestions = [];
      const oPatterns = this._suggestionPatterns[sFieldName];
            
      if (!oPatterns || !this.getModel('view').getProperty('/suggestions/userPreferences/showCompletionSuggestions')) {
        return aSuggestions;
      }
            
      // Generate completion suggestions based on enhanced patterns
      if (oPatterns.completionSuggestions) {
        oPatterns.completionSuggestions.forEach((oSuggestionConfig) => {
          const sSuggestionValue = typeof oSuggestionConfig === 'string' ? 
            oSuggestionConfig : oSuggestionConfig.value;
                    
          if (sCurrentValue.length > 0 && sSuggestionValue.toLowerCase().includes(sCurrentValue.toLowerCase())) {
            aSuggestions.push({
              type: 'completion',
              title: `Complete with: ${  sSuggestionValue}`,
              description: 'AI-suggested completion based on common patterns',
              action: typeof oSuggestionConfig === 'object' ? oSuggestionConfig.action : 'replace',
              value: sSuggestionValue,
              confidence: 0.8,
              icon: 'sap-icon://lightbulb',
              priority: 3,
              previewable: typeof oSuggestionConfig === 'object' ? oSuggestionConfig.previewable : true
            });
          }
        });
      }
            
      // Generate advanced suggestions (append, insert, modify)
      if (oPatterns.advancedSuggestions && sCurrentValue.length > 0) {
        oPatterns.advancedSuggestions.forEach((oAdvanced) => {
          const bTriggerMatch = !oAdvanced.trigger || 
                                       sCurrentValue.toLowerCase().includes(oAdvanced.trigger.toLowerCase());
                    
          if (bTriggerMatch) {
            aSuggestions.push({
              type: 'completion',
              title: oAdvanced.description || (`Add: ${  oAdvanced.value}`),
              description: `Enhanced completion with ${  oAdvanced.action  } operation`,
              action: oAdvanced.action,
              value: oAdvanced.value,
              insertPosition: oAdvanced.insertPosition,
              modifyPattern: oAdvanced.modifyPattern,
              confidence: 0.7,
              icon: 'sap-icon://add',
              priority: 3,
              previewable: true
            });
          }
        });
      }
            
      return aSuggestions;
    },
        
    _generateValidationSuggestions: function (sFieldName, sCurrentValue, oProjectData) {
      const aSuggestions = [];
      const oPatterns = this._suggestionPatterns[sFieldName];
            
      if (!this.getModel('view').getProperty('/suggestions/userPreferences/showValidationSuggestions')) {
        return aSuggestions;
      }
            
      // Generate validation-based suggestions
      const oValidationResult = this._validateField(sFieldName, sCurrentValue, oProjectData);
            
      if (!oValidationResult.isValid && oPatterns && oPatterns.validationTips) {
        oPatterns.validationTips.forEach((sTip, iIndex) => {
          aSuggestions.push({
            type: 'validation',
            title: 'Validation Tip',
            description: sTip,
            action: 'info',
            confidence: 0.9,
            icon: 'sap-icon://hint',
            priority: 1
          });
        });
      }
            
      return aSuggestions;
    },
        
    _generateOptimizationSuggestions: function (sFieldName, sCurrentValue, oProjectData) {
      const aSuggestions = [];
      const oPatterns = this._suggestionPatterns[sFieldName];
            
      if (!this.getModel('view').getProperty('/suggestions/userPreferences/showOptimizationSuggestions')) {
        return aSuggestions;
      }
            
      // Generate optimization suggestions based on project context
      if (oPatterns && oPatterns.optimizationSuggestions) {
        oPatterns.optimizationSuggestions.forEach((sSuggestion) => {
          aSuggestions.push({
            type: 'optimization',
            title: 'Optimization Suggestion',
            description: sSuggestion,
            action: 'info',
            confidence: 0.7,
            icon: 'sap-icon://business-objects-experience',
            priority: 2
          });
        });
      }
            
      // Generate contextual suggestions based on other field values
      if (sFieldName === 'budget' && oProjectData.startDate && oProjectData.endDate) {
        const iMonths = this._calculateProjectMonths(oProjectData.startDate, oProjectData.endDate);
        if (iMonths > 12) {
          aSuggestions.push({
            type: 'optimization',
            title: 'Long-term Project Budget',
            description: 'Consider phased budget allocation for projects over 12 months',
            action: 'info',
            confidence: 0.8,
            icon: 'sap-icon://calendar',
            priority: 2
          });
        }
      }
            
      return aSuggestions;
    },
        
    _prioritizeSuggestions: function (aSuggestions) {
      // Sort suggestions by priority (lower number = higher priority) and confidence
      return aSuggestions.sort((a, b) => {
        if (a.priority !== b.priority) {
          return a.priority - b.priority;
        }
        return b.confidence - a.confidence;
      });
    },
        
    _showSuggestionsPopover: function (sFieldName) {
      // Create and show suggestions popover
      if (!this._suggestionsPopover) {
        this._suggestionsPopover = sap.ui.xmlfragment('a2a.portal.view.fragments.SuggestionsPopover', this);
        this.getView().addDependent(this._suggestionsPopover);
      }
            
      const oControl = this.byId(sFieldName === 'name' ? 'projectNameInput' : 
        sFieldName === 'description' ? 'projectDescInput' :
          sFieldName === 'costCenter' ? 'costCenterInput' :
            sFieldName === 'budget' ? 'budgetInput' : null);
            
      if (oControl) {
        this._suggestionsPopover.openBy(oControl);
      }
    },
        
    onAcceptSuggestion: function (oEvent) {
      const oSuggestion = oEvent.getSource().getBindingContext('view').getObject();
      const sFieldName = this._getCurrentSuggestionField();
            
      // Enhanced acceptance handling with validation and rollback capability
      this._processSuggestionAcceptance(oSuggestion, sFieldName);
    },
        
    _getCurrentSuggestionField: function () {
      // Determine which field triggered the current suggestions
      const sSuggestionField = this.getModel('view').getProperty('/suggestions/activeSuggestion');
      if (!sSuggestionField) {
        // Fallback: detect from popover position or last focused field
        const oFocusedControl = document.activeElement;
        const sControlId = oFocusedControl ? oFocusedControl.id : '';
                
        if (sControlId.includes('projectNameInput')) {
          return 'name';
        }
        if (sControlId.includes('projectDescInput')) {
          return 'description';
        }
        if (sControlId.includes('costCenterInput')) {
          return 'costCenter';
        }
        if (sControlId.includes('budgetInput')) {
          return 'budget';
        }
        if (sControlId.includes('startDatePicker')) {
          return 'startDate';
        }
        if (sControlId.includes('endDatePicker')) {
          return 'endDate';
        }
      }
      return sSuggestionField;
    },
        
    _processSuggestionAcceptance: function (oSuggestion, sFieldName) {
      const oAcceptanceResult = {
        success: false,
        originalValue: null,
        newValue: null,
        fieldName: sFieldName,
        suggestion: oSuggestion,
        timestamp: new Date(),
        rollbackFunction: null
      };
            
      try {
        // Store original value for potential rollback
        const oControl = this._getFieldControl(sFieldName);
        if (!oControl) {
          throw new Error(`Target field control not found: ${  sFieldName}`);
        }
                
        oAcceptanceResult.originalValue = oControl.getValue();
                
        // Pre-acceptance validation
        this._validateSuggestionApplicability(oSuggestion, sFieldName, oControl);
                
        // Apply suggestion based on type
        switch (oSuggestion.action) {
        case 'replace':
          this._applySuggestionReplace(oSuggestion, oControl, oAcceptanceResult);
          break;
        case 'append':
          this._applySuggestionAppend(oSuggestion, oControl, oAcceptanceResult);
          break;
        case 'insert':
          this._applySuggestionInsert(oSuggestion, oControl, oAcceptanceResult);
          break;
        case 'modify':
          this._applySuggestionModify(oSuggestion, oControl, oAcceptanceResult);
          break;
        default:
          // Info-only suggestions don't modify field values
          oAcceptanceResult.success = true;
          oAcceptanceResult.newValue = oAcceptanceResult.originalValue;
        }
                
        // Post-application validation
        if (oSuggestion.action !== 'info') {
          this._validateAppliedSuggestion(oControl, oAcceptanceResult);
        }
                
        // Record successful acceptance
        this._recordSuggestionAcceptance(oAcceptanceResult);
                
        // Update UI and provide feedback
        this._updateUIAfterSuggestionAcceptance(oAcceptanceResult);
                
      } catch (oError) {
        // Handle acceptance failure
        this._handleSuggestionAcceptanceError(oError, oAcceptanceResult);
      }
    },
        
    _getFieldControl: function (sFieldName) {
      const mFieldControlMap = {
        'name': 'projectNameInput',
        'description': 'projectDescInput',
        'costCenter': 'costCenterInput',
        'budget': 'budgetInput',
        'startDate': 'startDatePicker',
        'endDate': 'endDatePicker'
      };
            
      const sControlId = mFieldControlMap[sFieldName];
      return sControlId ? this.byId(sControlId) : null;
    },
        
    _validateSuggestionApplicability: function (oSuggestion, sFieldName, oControl) {
      // Check field editability
      if (!oControl.getEditable || !oControl.getEditable()) {
        throw new Error(`Field is not editable: ${  sFieldName}`);
      }
            
      // Check suggestion compatibility with field type
      const sFieldType = this._getFieldType(oControl);
      if (!this._isSuggestionCompatible(oSuggestion, sFieldType)) {
        throw new Error(`Suggestion incompatible with field type: ${  sFieldType}`);
      }
            
      // Check field-specific constraints
      this._validateFieldConstraints(oSuggestion, sFieldName, oControl);
    },
        
    _getFieldType: function (oControl) {
      const sControlName = oControl.getMetadata().getName();
      if (sControlName.includes('Input')) {
        return 'text';
      }
      if (sControlName.includes('TextArea')) {
        return 'textarea';
      }
      if (sControlName.includes('DatePicker')) {
        return 'date';
      }
      if (sControlName.includes('ComboBox')) {
        return 'select';
      }
      return 'unknown';
    },
        
    _isSuggestionCompatible: function (oSuggestion, sFieldType) {
      const mCompatibilityMatrix = {
        'text': ['replace', 'append', 'insert', 'modify', 'info'],
        'textarea': ['replace', 'append', 'insert', 'modify', 'info'],
        'date': ['replace', 'info'],
        'select': ['replace', 'info'],
        'unknown': ['info']
      };
            
      const aCompatibleActions = mCompatibilityMatrix[sFieldType] || ['info'];
      return aCompatibleActions.includes(oSuggestion.action);
    },
        
    _validateFieldConstraints: function (oSuggestion, sFieldName, oControl) {
      // Field-specific validation rules
      switch (sFieldName) {
      case 'name':
        if (oSuggestion.value && oSuggestion.value.length > 100) {
          throw new Error('Project name exceeds maximum length (100 characters)');
        }
        break;
      case 'costCenter':
        if (oSuggestion.value && !/^CC-[A-Z]{2,3}-\d{3}$/.test(oSuggestion.value)) {
          throw new Error('Cost center format invalid. Expected: CC-XXX-###');
        }
        break;
      case 'budget':
        if (oSuggestion.value && (isNaN(oSuggestion.value) || parseFloat(oSuggestion.value) < 0)) {
          throw new Error('Budget must be a positive number');
        }
        break;
      }
    },
        
    _applySuggestionReplace: function (oSuggestion, oControl, oResult) {
      const sNewValue = oSuggestion.value || '';
      oControl.setValue(sNewValue);
      oControl.fireChange({ value: sNewValue });
            
      oResult.newValue = sNewValue;
      oResult.success = true;
      oResult.rollbackFunction = function () {
        oControl.setValue(oResult.originalValue);
        oControl.fireChange({ value: oResult.originalValue });
      };
    },
        
    _applySuggestionAppend: function (oSuggestion, oControl, oResult) {
      const sCurrentValue = oResult.originalValue || '';
      const sAppendValue = oSuggestion.value || '';
      const sNewValue = sCurrentValue + sAppendValue;
            
      oControl.setValue(sNewValue);
      oControl.fireChange({ value: sNewValue });
            
      oResult.newValue = sNewValue;
      oResult.success = true;
      oResult.rollbackFunction = function () {
        oControl.setValue(oResult.originalValue);
        oControl.fireChange({ value: oResult.originalValue });
      };
    },
        
    _applySuggestionInsert: function (oSuggestion, oControl, oResult) {
      const sCurrentValue = oResult.originalValue || '';
      const iInsertPosition = oSuggestion.insertPosition || sCurrentValue.length;
      const sInsertValue = oSuggestion.value || '';
            
      const sNewValue = sCurrentValue.substring(0, iInsertPosition) + 
                           sInsertValue + 
                           sCurrentValue.substring(iInsertPosition);
            
      oControl.setValue(sNewValue);
      oControl.fireChange({ value: sNewValue });
            
      oResult.newValue = sNewValue;
      oResult.success = true;
      oResult.rollbackFunction = function () {
        oControl.setValue(oResult.originalValue);
        oControl.fireChange({ value: oResult.originalValue });
      };
    },
        
    _applySuggestionModify: function (oSuggestion, oControl, oResult) {
      const sCurrentValue = oResult.originalValue || '';
      const sPattern = oSuggestion.modifyPattern || '';
      const sReplacement = oSuggestion.value || '';
            
      const sNewValue = sCurrentValue.replace(new RegExp(sPattern, 'g'), sReplacement);
            
      oControl.setValue(sNewValue);
      oControl.fireChange({ value: sNewValue });
            
      oResult.newValue = sNewValue;
      oResult.success = true;
      oResult.rollbackFunction = function () {
        oControl.setValue(oResult.originalValue);
        oControl.fireChange({ value: oResult.originalValue });
      };
    },
        
    _validateAppliedSuggestion: function (oControl, oResult) {
      // Run field validation on the new value
      const sFieldName = oResult.fieldName;
      const sNewValue = oResult.newValue;
      const oProjectData = this.getModel('project').getData();
            
      const oValidationResult = this._validateField(sFieldName, sNewValue, oProjectData);
      if (!oValidationResult.isValid) {
        // Log validation issues but don't fail the acceptance
        console.warn('Suggestion applied but field validation failed:', oValidationResult.errors);
        oResult.validationWarnings = oValidationResult.errors;
      }
    },
        
    _recordSuggestionAcceptance: function (oResult) {
      // Enhanced feedback recording with detailed acceptance data
      const oFeedback = this.getModel('view').getProperty('/suggestions/feedback');
            
      oFeedback.accepted = oFeedback.accepted || [];
      oFeedback.applied = oFeedback.applied || [];
            
      const oAcceptanceRecord = {
        suggestion: oResult.suggestion,
        fieldName: oResult.fieldName,
        originalValue: oResult.originalValue,
        newValue: oResult.newValue,
        timestamp: oResult.timestamp,
        success: oResult.success,
        validationWarnings: oResult.validationWarnings || [],
        userAgent: navigator.userAgent,
        sessionId: this._getSessionId()
      };
            
      oFeedback.accepted.push(oAcceptanceRecord);
      if (oResult.success) {
        oFeedback.applied.push(oAcceptanceRecord);
      }
            
      this.getModel('view').setProperty('/suggestions/feedback', oFeedback);
            
      // Send to backend learning service
      this._sendAcceptanceFeedbackToBackend(oAcceptanceRecord);
    },
        
    _updateUIAfterSuggestionAcceptance: function (oResult) {
      // Update real-time validation
      this._updateLiveValidationIndicator();
            
      // Close suggestions popover
      this._closeSuggestionsPopover();
            
      // Show success feedback
      const sMessage = `Suggestion applied: ${  oResult.suggestion.title}`;
      if (oResult.validationWarnings && oResult.validationWarnings.length > 0) {
        sMessage += ` (validation warnings: ${  oResult.validationWarnings.length  })`;
      }
            
      MessageToast.show(sMessage);
            
      // Offer undo capability for replace/modify actions
      if (oResult.rollbackFunction && ['replace', 'modify'].includes(oResult.suggestion.action)) {
        this._offerSuggestionUndo(oResult);
      }
    },
        
    _handleSuggestionAcceptanceError: function (oError, oResult) {
      console.error('Suggestion acceptance failed:', oError);
            
      // Record failure for learning
      const oFeedback = this.getModel('view').getProperty('/suggestions/feedback');
      oFeedback.failed = oFeedback.failed || [];
            
      oFeedback.failed.push({
        suggestion: oResult.suggestion,
        fieldName: oResult.fieldName,
        error: oError.message,
        timestamp: oResult.timestamp
      });
            
      this.getModel('view').setProperty('/suggestions/feedback', oFeedback);
            
      // Show error message to user
      MessageBox.error(`Failed to apply suggestion: ${  oError.message}`, {
        title: 'Suggestion Error',
        actions: [MessageBox.Action.OK]
      });
    },
        
    _offerSuggestionUndo: function (oResult) {
      // Show undo option in a non-intrusive way
      const oUndoBar = new sap.m.MessageStrip({
        text: 'Suggestion applied. Click to undo.',
        type: 'Success',
        showCloseButton: true,
        class: 'sapUiSmallMargin a2a-suggestion-undo-bar'
      });
            
      oUndoBar.attachPress(() => {
        if (oResult.rollbackFunction) {
          oResult.rollbackFunction();
          MessageToast.show('Suggestion undone');
        }
        oUndoBar.destroy();
      });
            
      // Add to page temporarily
      const oPage = this.byId('objectPageLayout');
      if (oPage) {
        oPage.addHeaderContent(oUndoBar);
                
        // Auto-remove after 10 seconds
        setTimeout(() => {
          if (!oUndoBar.bIsDestroyed) {
            oUndoBar.destroy();
          }
        }, 10000);
      }
    },
        
    _sendAcceptanceFeedbackToBackend: function (oRecord) {
      // Enhanced backend communication for acceptance tracking
      $.ajax({
        url: '/api/suggestions/acceptance',
        method: 'POST',
        data: JSON.stringify(oRecord),
        contentType: 'application/json',
        headers: {
          'X-Session-ID': this._getSessionId(),
          'X-User-ID': this.getModel('user').getProperty('/id')
        },
        success: function (data) {
          // eslint-disable-next-line no-console
          // eslint-disable-next-line no-console
          console.log('Acceptance feedback recorded:', data);
        },
        error: function (xhr, status, error) {
          console.warn('Failed to record acceptance feedback:', error);
        }
      });
    },
        
    _getSessionId: function () {
      // Generate or retrieve session identifier
      if (!this._sessionId) {
        this._sessionId = `session_${  Date.now()  }_${  Math.random().toString(36).substr(2, 9)}`;
      }
      return this._sessionId;
    },
        
    onRejectSuggestion: function (oEvent) {
      const oSuggestion = oEvent.getSource().getBindingContext('view').getObject();
      const sFieldName = this._getCurrentSuggestionField();
            
      // Enhanced rejection handling with feedback collection
      this._processSuggestionRejection(oSuggestion, sFieldName, oEvent);
    },
        
    _processSuggestionRejection: function (oSuggestion, sFieldName, oEvent) {
      const oRejectionContext = {
        suggestion: oSuggestion,
        fieldName: sFieldName,
        timestamp: new Date(),
        userContext: this._getCurrentUserContext(),
        rejectionReason: null,
        sessionId: this._getSessionId(),
        fieldValue: this._getCurrentFieldValue(sFieldName),
        alternativesConsidered: this._getAlternativeSuggestions(oSuggestion)
      };
            
      // Determine rejection reason and show appropriate feedback
      if (oSuggestion.type === 'validation' && oSuggestion.action === 'info') {
        this._handleInfoSuggestionRejection(oRejectionContext);
      } else {
        this._handleActionableSuggestionRejection(oRejectionContext);
      }
    },
        
    _getCurrentUserContext: function () {
      return {
        projectStatus: this.getModel('project').getProperty('/status'),
        editMode: this.getModel('view').getProperty('/editMode'),
        userRole: this.getModel('view').getProperty('/fieldEditability/userRole'),
        suggestionsEnabled: this.getModel('view').getProperty('/suggestions/enabled'),
        completedFields: this._getCompletedFieldsCount(),
        timeInEditMode: this._calculateEditModeTime()
      };
    },
        
    _getCurrentFieldValue: function (sFieldName) {
      const oControl = this._getFieldControl(sFieldName);
      return oControl ? oControl.getValue() : null;
    },
        
    _getAlternativeSuggestions: function (oRejectedSuggestion) {
      const aSuggestions = this.getModel('view').getProperty('/suggestions/currentSuggestions');
      return aSuggestions.filter((oSugg) => {
        return oSugg !== oRejectedSuggestion;
      }).map((oSugg) => {
        return {
          type: oSugg.type,
          action: oSugg.action,
          confidence: oSugg.confidence
        };
      });
    },
        
    _handleInfoSuggestionRejection: function (oContext) {
      // For info-only suggestions, record as "acknowledged" rather than "rejected"
      oContext.rejectionReason = 'acknowledged';
      this._recordEnhancedRejection(oContext);
      this._removeSuggestionFromList(oContext.suggestion);
            
      MessageToast.show('Got it! Tip acknowledged');
            
      // Don't show this type of suggestion again for a while
      this._temporarilySupressSuggestionType(oContext.suggestion.type, oContext.fieldName);
    },
        
    _handleActionableSuggestionRejection: function (oContext) {
      const oSuggestion = oContext.suggestion;
            
      // Show rejection reason dialog for actionable suggestions
      this._showRejectionReasonDialog(oContext, (sReason, bBlockSimilar) => {
        oContext.rejectionReason = sReason;
        oContext.blockSimilar = bBlockSimilar;
                
        this._recordEnhancedRejection(oContext);
        this._removeSuggestionFromList(oSuggestion);
                
        if (bBlockSimilar) {
          this._addToRejectionPatterns(oSuggestion, sReason);
        }
                
        this._showRejectionFeedback(sReason, bBlockSimilar);
      });
    },
        
    _showRejectionReasonDialog: function (oContext, fnCallback) {
      const oSuggestion = oContext.suggestion;
            
      const oDialog = new sap.m.Dialog({
        title: "Why don't you want this suggestion?",
        type: 'Message',
        content: [
          new sap.m.VBox({
            class: 'sapUiMediumMargin',
            items: [
              new sap.m.Text({
                text: "Help us improve suggestions by telling us why you're dismissing:",
                class: 'sapUiSmallMarginBottom'
              }),
              new sap.m.Text({
                text: `"${  oSuggestion.title  }"`,
                class: 'sapUiSmallMarginBottom sapMTextBold'
              }),
              new sap.m.RadioButtonGroup({
                id: 'rejectionReasonGroup',
                columns: 1,
                buttons: [
                  new sap.m.RadioButton({
                    text: 'Not relevant for this field',
                    key: 'not_relevant'
                  }),
                  new sap.m.RadioButton({
                    text: 'Suggestion is incorrect or inappropriate',
                    key: 'incorrect'
                  }),
                  new sap.m.RadioButton({
                    text: 'I prefer a different approach',
                    key: 'different_approach'
                  }),
                  new sap.m.RadioButton({
                    text: 'Too many suggestions (overwhelming)',
                    key: 'too_many'
                  }),
                  new sap.m.RadioButton({
                    text: 'Suggestion appears at wrong time',
                    key: 'wrong_timing'
                  }),
                  new sap.m.RadioButton({
                    text: 'Other reason',
                    key: 'other'
                  })
                ]
              }),
              new sap.m.CheckBox({
                id: 'blockSimilarCheckbox',
                text: "Don't show similar suggestions",
                class: 'sapUiMediumMarginTop'
              })
            ]
          })
        ],
        buttons: [
          new sap.m.Button({
            text: 'Submit Feedback',
            type: 'Emphasized',
            press: function () {
              const sSelectedReason = sap.ui.getCore().byId('rejectionReasonGroup').getSelectedButton()?.getKey() || 'no_reason';
              const bBlockSimilar = sap.ui.getCore().byId('blockSimilarCheckbox').getSelected();
                            
              oDialog.close();
              fnCallback(sSelectedReason, bBlockSimilar);
            }
          }),
          new sap.m.Button({
            text: 'Just Dismiss',
            type: 'Default',
            press: function () {
              oDialog.close();
              fnCallback('dismissed_without_reason', false);
            }
          })
        ]
      });
            
      oDialog.open();
    },
        
    _recordEnhancedRejection: function (oContext) {
      const oFeedback = this.getModel('view').getProperty('/suggestions/feedback');
      oFeedback.rejected = oFeedback.rejected || [];
            
      const oRejectionRecord = {
        suggestion: oContext.suggestion,
        fieldName: oContext.fieldName,
        rejectionReason: oContext.rejectionReason,
        fieldValue: oContext.fieldValue,
        userContext: oContext.userContext,
        alternativesAvailable: oContext.alternativesConsidered.length,
        timestamp: oContext.timestamp,
        sessionId: oContext.sessionId,
        blockSimilar: oContext.blockSimilar || false
      };
            
      oFeedback.rejected.push(oRejectionRecord);
      this.getModel('view').setProperty('/suggestions/feedback', oFeedback);
            
      // Send enhanced feedback to backend
      this._sendRejectionFeedbackToBackend(oRejectionRecord);
            
      // Update user preferences based on rejection patterns
      this._updateUserPreferencesFromRejection(oRejectionRecord);
    },
        
    _removeSuggestionFromList: function (oSuggestion) {
      const aSuggestions = this.getModel('view').getProperty('/suggestions/currentSuggestions');
      const iIndex = aSuggestions.indexOf(oSuggestion);
      if (iIndex > -1) {
        aSuggestions.splice(iIndex, 1);
        this.getModel('view').setProperty('/suggestions/currentSuggestions', aSuggestions);
      }
            
      // Close popover if no more suggestions
      if (aSuggestions.length === 0) {
        this._closeSuggestionsPopover();
      }
    },
        
    _temporarilySupressSuggestionType: function (sType, sFieldName) {
      const oSuppressionData = this.getModel('view').getProperty('/suggestions/temporarySuppression') || {};
      const sKey = `${sFieldName  }_${  sType}`;
            
      oSuppressionData[sKey] = {
        suppressedUntil: Date.now() + (5 * 60 * 1000), // 5 minutes
        reason: 'user_acknowledged'
      };
            
      this.getModel('view').setProperty('/suggestions/temporarySuppression', oSuppressionData);
    },
        
    _addToRejectionPatterns: function (oSuggestion, sReason) {
      const oPatterns = this.getModel('view').getProperty('/suggestions/rejectionPatterns') || {};
            
      const sPatternKey = `${oSuggestion.type  }_${  oSuggestion.action}`;
      if (!oPatterns[sPatternKey]) {
        oPatterns[sPatternKey] = [];
      }
            
      oPatterns[sPatternKey].push({
        value: oSuggestion.value,
        reason: sReason,
        timestamp: new Date(),
        fieldName: this._getCurrentSuggestionField()
      });
            
      this.getModel('view').setProperty('/suggestions/rejectionPatterns', oPatterns);
    },
        
    _showRejectionFeedback: function (sReason, bBlockSimilar) {
      let sMessage;
            
      switch (sReason) {
      case 'not_relevant':
        sMessage = "Thanks! We'll improve relevance for this field type";
        break;
      case 'incorrect':
        sMessage = "Feedback noted - we'll review this suggestion pattern";
        break;
      case 'different_approach':
        sMessage = "Got it! We'll learn from your preferred approach";
        break;
      case 'too_many':
        sMessage = "We'll reduce suggestion frequency for you";
        break;
      case 'wrong_timing':
        sMessage = "Timing noted - we'll improve suggestion triggers";
        break;
      default:
        sMessage = 'Suggestion dismissed';
      }
            
      if (bBlockSimilar) {
        sMessage += ' (similar suggestions blocked)';
      }
            
      MessageToast.show(sMessage);
    },
        
    _sendRejectionFeedbackToBackend: function (oRecord) {
      // Enhanced backend communication for rejection tracking
      $.ajax({
        url: '/api/suggestions/rejection',
        method: 'POST',
        data: JSON.stringify(oRecord),
        contentType: 'application/json',
        headers: {
          'X-Session-ID': this._getSessionId(),
          'X-User-ID': this.getModel('user').getProperty('/id'),
          'X-Field-Context': oRecord.fieldName
        },
        success: function (data) {
          // eslint-disable-next-line no-console
          // eslint-disable-next-line no-console
          console.log('Rejection feedback recorded:', data);
          // Backend might return improved suggestions based on feedback
          if (data.improvedSuggestions && data.improvedSuggestions.length > 0) {
            this._incorporateImprovedSuggestions(data.improvedSuggestions);
          }
        }.bind(this),
        error: function (xhr, status, error) {
          console.warn('Failed to record rejection feedback:', error);
        }
      });
    },
        
    _updateUserPreferencesFromRejection: function (oRecord) {
      const oPrefs = this.getModel('view').getProperty('/suggestions/userPreferences');
            
      // Adjust preferences based on rejection patterns
      switch (oRecord.rejectionReason) {
      case 'too_many':
        // Reduce suggestion frequency
        this.getModel('view').setProperty('/suggestions/showDelay', 
          Math.min(this.getModel('view').getProperty('/suggestions/showDelay') + 500, 3000));
        this.getModel('view').setProperty('/suggestions/maxSuggestions', 
          Math.max(this.getModel('view').getProperty('/suggestions/maxSuggestions') - 1, 1));
        break;
      case 'wrong_timing':
        // Increase delay for this user
        this.getModel('view').setProperty('/suggestions/showDelay', 
          Math.min(this.getModel('view').getProperty('/suggestions/showDelay') + 250, 2500));
        break;
      case 'not_relevant':
        // Reduce confidence threshold for this suggestion type
        const sType = oRecord.suggestion.type;
        if (sType === 'completion') {
          oPrefs.showCompletionSuggestions = false;
        } else if (sType === 'optimization') {
          oPrefs.showOptimizationSuggestions = false;
        }
        break;
      }
            
      this.getModel('view').setProperty('/suggestions/userPreferences', oPrefs);
    },
        
    _incorporateImprovedSuggestions: function (aImprovedSuggestions) {
      // Backend provided better suggestions based on rejection feedback
      const aCurrent = this.getModel('view').getProperty('/suggestions/currentSuggestions');
            
      aImprovedSuggestions.forEach((oImproved) => {
        oImproved.confidence = oImproved.confidence || 0.9; // High confidence for improved suggestions
        oImproved.priority = 1; // High priority
        oImproved.icon = 'sap-icon://learning-assistant';
        oImproved.improved = true;
        aCurrent.push(oImproved);
      });
            
      // Re-prioritize suggestions
      aCurrent = this._prioritizeSuggestions(aCurrent);
      this.getModel('view').setProperty('/suggestions/currentSuggestions', aCurrent);
    },
        
    _getCompletedFieldsCount: function () {
      const oProject = this.getModel('project').getData();
      const aFieldNames = ['name', 'description', 'startDate', 'endDate', 'budget', 'costCenter'];
            
      return aFieldNames.filter((sField) => {
        return oProject[sField] && oProject[sField].toString().trim().length > 0;
      }).length;
    },
        
    // === LEARNING FEEDBACK SYSTEM ===
        
    onViewLearningAnalytics: function () {
      const oLearningData = this._generateLearningAnalytics();
            
      if (!this._oLearningAnalyticsDialog) {
        this._oLearningAnalyticsDialog = sap.ui.xmlfragment('a2a.portal.view.fragments.LearningAnalyticsDialog', this);
        this.getView().addDependent(this._oLearningAnalyticsDialog);
      }
            
      // Update learning analytics model
      this.getModel('view').setProperty('/learningAnalytics', oLearningData);
      this._oLearningAnalyticsDialog.open();
    },
        
    _generateLearningAnalytics: function () {
      const oFeedback = this.getModel('view').getProperty('/suggestions/feedback') || { accepted: [], rejected: [], applied: [] };
      const oPrefs = this.getModel('view').getProperty('/suggestions/userPreferences') || {};
            
      // Calculate learning metrics
      const iTotalInteractions = (oFeedback.accepted || []).length + (oFeedback.rejected || []).length;
      const fAcceptanceRate = iTotalInteractions > 0 ? ((oFeedback.accepted || []).length / iTotalInteractions) * 100 : 0;
            
      // Analyze rejection patterns
      const oRejectionAnalysis = this._analyzeRejectionPatterns(oFeedback.rejected || []);
            
      // Analyze acceptance patterns  
      const oAcceptanceAnalysis = this._analyzeAcceptancePatterns(oFeedback.accepted || []);
            
      // Generate improvement recommendations
      const aImprovements = this._generateImprovementRecommendations(oRejectionAnalysis, oAcceptanceAnalysis);
            
      // User engagement metrics
      const oEngagement = this._calculateEngagementMetrics(oFeedback);
            
      // Learning evolution over time
      const oEvolution = this._analyzeLearningEvolution(oFeedback);
            
      return {
        totalInteractions: iTotalInteractions,
        acceptanceRate: Math.round(fAcceptanceRate),
        rejectionRate: Math.round(100 - fAcceptanceRate),
        appliedSuggestions: (oFeedback.applied || []).length,
        rejectionAnalysis: oRejectionAnalysis,
        acceptanceAnalysis: oAcceptanceAnalysis,
        improvements: aImprovements,
        engagement: oEngagement,
        evolution: oEvolution,
        preferences: oPrefs,
        confidence: this._calculateSystemConfidence(oFeedback),
        learningScore: this._calculateLearningScore(oFeedback, oEngagement),
        lastUpdated: new Date()
      };
    },
        
    _analyzeRejectionPatterns: function (aRejected) {
      const oReasons = {};
      const oFieldTypes = {};
      const oSuggestionTypes = {};
      const oTimePatterns = {};
            
      aRejected.forEach((oRejection) => {
        // Reason analysis
        oReasons[oRejection.rejectionReason] = (oReasons[oRejection.rejectionReason] || 0) + 1;
                
        // Field type analysis
        oFieldTypes[oRejection.fieldName] = (oFieldTypes[oRejection.fieldName] || 0) + 1;
                
        // Suggestion type analysis
        const sType = oRejection.suggestion ? oRejection.suggestion.type : 'unknown';
        oSuggestionTypes[sType] = (oSuggestionTypes[sType] || 0) + 1;
                
        // Time pattern analysis
        if (oRejection.timestamp) {
          const iHour = new Date(oRejection.timestamp).getHours();
          const sTimeSlot = iHour < 12 ? 'morning' : iHour < 18 ? 'afternoon' : 'evening';
          oTimePatterns[sTimeSlot] = (oTimePatterns[sTimeSlot] || 0) + 1;
        }
      });
            
      return {
        totalRejections: aRejected.length,
        reasonBreakdown: oReasons,
        fieldBreakdown: oFieldTypes,
        typeBreakdown: oSuggestionTypes,
        timePatterns: oTimePatterns,
        mostRejectedReason: this._findMostFrequent(oReasons),
        mostProblematicField: this._findMostFrequent(oFieldTypes),
        mostRejectedType: this._findMostFrequent(oSuggestionTypes)
      };
    },
        
    _analyzeAcceptancePatterns: function (aAccepted) {
      const oFieldTypes = {};
      const oSuggestionTypes = {};
      const oTimePatterns = {};
      const oConfidenceRanges = { low: 0, medium: 0, high: 0 };
            
      aAccepted.forEach((oAcceptance) => {
        // Field type analysis
        oFieldTypes[oAcceptance.fieldName] = (oFieldTypes[oAcceptance.fieldName] || 0) + 1;
                
        // Suggestion type analysis
        const sType = oAcceptance.suggestion ? oAcceptance.suggestion.type : 'unknown';
        oSuggestionTypes[sType] = (oSuggestionTypes[sType] || 0) + 1;
                
        // Time pattern analysis
        if (oAcceptance.timestamp) {
          const iHour = new Date(oAcceptance.timestamp).getHours();
          const sTimeSlot = iHour < 12 ? 'morning' : iHour < 18 ? 'afternoon' : 'evening';
          oTimePatterns[sTimeSlot] = (oTimePatterns[sTimeSlot] || 0) + 1;
        }
                
        // Confidence analysis
        if (oAcceptance.suggestion && oAcceptance.suggestion.confidence) {
          const fConf = oAcceptance.suggestion.confidence;
          if (fConf < 0.6) {
            oConfidenceRanges.low++;
          } else if (fConf < 0.8) {
            oConfidenceRanges.medium++;
          } else {
            oConfidenceRanges.high++;
          }
        }
      });
            
      return {
        totalAcceptances: aAccepted.length,
        fieldBreakdown: oFieldTypes,
        typeBreakdown: oSuggestionTypes,
        timePatterns: oTimePatterns,
        confidenceRanges: oConfidenceRanges,
        mostAcceptedField: this._findMostFrequent(oFieldTypes),
        mostAcceptedType: this._findMostFrequent(oSuggestionTypes),
        preferredTime: this._findMostFrequent(oTimePatterns)
      };
    },
        
    _generateImprovementRecommendations: function (oRejection, oAcceptance) {
      const aRecommendations = [];
            
      // Analyze rejection reasons for improvements
      if (oRejection.mostRejectedReason === 'not_relevant') {
        aRecommendations.push({
          category: 'relevance',
          priority: 'high',
          title: 'Improve Suggestion Relevance',
          description: 'Too many suggestions marked as not relevant. AI should focus on context-specific recommendations.',
          actionable: true,
          impact: 'high'
        });
      }
            
      if (oRejection.mostRejectedReason === 'too_many') {
        aRecommendations.push({
          category: 'frequency',
          priority: 'high',
          title: 'Reduce Suggestion Frequency',
          description: 'User finds suggestions overwhelming. Implement intelligent throttling.',
          actionable: true,
          impact: 'medium'
        });
      }
            
      if (oRejection.mostRejectedReason === 'wrong_timing') {
        aRecommendations.push({
          category: 'timing',
          priority: 'medium',
          title: 'Improve Timing Intelligence',
          description: 'Suggestions shown at inappropriate times. Need better context awareness.',
          actionable: true,
          impact: 'medium'
        });
      }
            
      // Analyze acceptance patterns for reinforcement
      if (oAcceptance.mostAcceptedType && oAcceptance.typeBreakdown[oAcceptance.mostAcceptedType] > 3) {
        aRecommendations.push({
          category: 'reinforcement',
          priority: 'low',
          title: `Increase ${  oAcceptance.mostAcceptedType  } Suggestions`,
          description: `User frequently accepts ${  oAcceptance.mostAcceptedType  } type suggestions. Prioritize these.`,
          actionable: true,
          impact: 'medium'
        });
      }
            
      // Field-specific recommendations
      if (oRejection.mostProblematicField) {
        aRecommendations.push({
          category: 'field-specific',
          priority: 'medium', 
          title: `Review ${  oRejection.mostProblematicField  } Field Logic`,
          description: `High rejection rate for ${  oRejection.mostProblematicField  } field suggestions.`,
          actionable: true,
          impact: 'low'
        });
      }
            
      return aRecommendations;
    },
        
    _calculateEngagementMetrics: function (oFeedback) {
      const iTotalInteractions = (oFeedback.accepted || []).length + (oFeedback.rejected || []).length;
      const iAppliedSuggestions = (oFeedback.applied || []).length;
            
      // Calculate session-based metrics
      const oSessions = this._analyzeUserSessions(oFeedback);
            
      return {
        totalInteractions: iTotalInteractions,
        appliedSuggestions: iAppliedSuggestions,
        sessions: oSessions,
        avgInteractionsPerSession: oSessions.totalSessions > 0 ? Math.round(iTotalInteractions / oSessions.totalSessions) : 0,
        engagementLevel: this._calculateEngagementLevel(iTotalInteractions, iAppliedSuggestions),
        userType: this._determineUserType(oFeedback)
      };
    },
        
    _analyzeUserSessions: function (oFeedback) {
      const oSessions = {};
      const allInteractions = [].concat(oFeedback.accepted || [], oFeedback.rejected || []);
            
      allInteractions.forEach((oInteraction) => {
        if (oInteraction.sessionId) {
          if (!oSessions[oInteraction.sessionId]) {
            oSessions[oInteraction.sessionId] = {
              interactions: 0,
              startTime: oInteraction.timestamp,
              endTime: oInteraction.timestamp
            };
          }
          oSessions[oInteraction.sessionId].interactions++;
          if (oInteraction.timestamp > oSessions[oInteraction.sessionId].endTime) {
            oSessions[oInteraction.sessionId].endTime = oInteraction.timestamp;
          }
        }
      });
            
      return {
        totalSessions: Object.keys(oSessions).length,
        sessionDetails: oSessions,
        avgSessionLength: this._calculateAvgSessionLength(oSessions)
      };
    },
        
    _analyzeLearningEvolution: function (oFeedback) {
      const allInteractions = [].concat(
        (oFeedback.accepted || []).map((item) => {
          return Object.assign({}, item, {type: 'accepted'}); 
        }),
        (oFeedback.rejected || []).map((item) => {
          return Object.assign({}, item, {type: 'rejected'}); 
        })
      );
            
      // Sort by timestamp
      allInteractions.sort((a, b) => {
        return new Date(a.timestamp) - new Date(b.timestamp);
      });
            
      const aEvolution = [];
      const iRunningAccepted = 0;
      const iRunningRejected = 0;
            
      allInteractions.forEach((oInteraction, iIndex) => {
        if (oInteraction.type === 'accepted') {
          iRunningAccepted++;
        } else {
          iRunningRejected++;
        }
                
        if ((iIndex + 1) % 5 === 0) { // Every 5 interactions
          const iTotalSoFar = iRunningAccepted + iRunningRejected;
          aEvolution.push({
            interactionNumber: iIndex + 1,
            acceptanceRate: Math.round((iRunningAccepted / iTotalSoFar) * 100),
            timestamp: oInteraction.timestamp,
            trend: aEvolution.length > 0 ? 
              (Math.round((iRunningAccepted / iTotalSoFar) * 100) > aEvolution[aEvolution.length - 1].acceptanceRate ? 'improving' : 'declining') :
              'neutral'
          });
        }
      });
            
      return {
        dataPoints: aEvolution,
        overallTrend: this._calculateOverallTrend(aEvolution),
        improvementPeriods: this._identifyImprovementPeriods(aEvolution),
        currentPerformance: aEvolution.length > 0 ? aEvolution[aEvolution.length - 1].acceptanceRate : 0
      };
    },
        
    _calculateSystemConfidence: function (oFeedback) {
      const iTotalInteractions = (oFeedback.accepted || []).length + (oFeedback.rejected || []).length;
      const fAcceptanceRate = iTotalInteractions > 0 ? ((oFeedback.accepted || []).length / iTotalInteractions) : 0;
            
      // Confidence increases with more interactions and higher acceptance rate
      const fInteractionWeight = Math.min(iTotalInteractions / 50, 1); // Max weight at 50 interactions
      const fAcceptanceWeight = fAcceptanceRate;
            
      return Math.round(((fInteractionWeight * 0.3) + (fAcceptanceWeight * 0.7)) * 100);
    },
        
    _calculateLearningScore: function (oFeedback, oEngagement) {
      const fAcceptanceRate = ((oFeedback.accepted || []).length) / 
                Math.max(((oFeedback.accepted || []).length + (oFeedback.rejected || []).length), 1);
      const fEngagementScore = Math.min(oEngagement.totalInteractions / 20, 1); // Max at 20 interactions
      const fApplicationScore = Math.min((oFeedback.applied || []).length / 10, 1); // Max at 10 applications
            
      return Math.round(((fAcceptanceRate * 0.4) + (fEngagementScore * 0.3) + (fApplicationScore * 0.3)) * 100);
    },
        
    _findMostFrequent: function (oFrequencyMap) {
      const sMaxKey = null;
      const iMaxCount = 0;
            
      for (const sKey in oFrequencyMap) {
        if (oFrequencyMap[sKey] > iMaxCount) {
          iMaxCount = oFrequencyMap[sKey];
          sMaxKey = sKey;
        }
      }
            
      return sMaxKey;
    },
        
    _calculateEngagementLevel: function (iTotalInteractions, iAppliedSuggestions) {
      if (iTotalInteractions === 0) {
        return 'none';
      }
      if (iTotalInteractions < 5) {
        return 'low';
      }
      if (iTotalInteractions < 15) {
        return 'moderate';
      }
      if (iTotalInteractions < 30) {
        return 'high';
      }
      return 'expert';
    },
        
    _determineUserType: function (oFeedback) {
      const iTotalInteractions = (oFeedback.accepted || []).length + (oFeedback.rejected || []).length;
      const fAcceptanceRate = iTotalInteractions > 0 ? ((oFeedback.accepted || []).length / iTotalInteractions) : 0;
            
      if (fAcceptanceRate > 0.8) {
        return 'accepter';
      }
      if (fAcceptanceRate < 0.3) {
        return 'skeptical';
      }
      return 'balanced';
    },
        
    _calculateAvgSessionLength: function (oSessions) {
      const iTotalLength = 0;
      const iSessionCount = 0;
            
      for (const sSessionId in oSessions) {
        const oSession = oSessions[sSessionId];
        if (oSession.startTime && oSession.endTime) {
          iTotalLength += new Date(oSession.endTime) - new Date(oSession.startTime);
          iSessionCount++;
        }
      }
            
      return iSessionCount > 0 ? Math.round(iTotalLength / iSessionCount / 1000 / 60) : 0; // Minutes
    },
        
    _calculateOverallTrend: function (aEvolution) {
      if (aEvolution.length < 2) {
        return 'insufficient_data';
      }
            
      const fFirstRate = aEvolution[0].acceptanceRate;
      const fLastRate = aEvolution[aEvolution.length - 1].acceptanceRate;
      const fDifference = fLastRate - fFirstRate;
            
      if (fDifference > 10) {
        return 'improving';
      }
      if (fDifference < -10) {
        return 'declining';
      }
      return 'stable';
    },
        
    _identifyImprovementPeriods: function (aEvolution) {
      const aPeriods = [];
      const iCurrentStart = null;
            
      for (let i = 1; i < aEvolution.length; i++) {
        if (aEvolution[i].trend === 'improving' && iCurrentStart === null) {
          iCurrentStart = i - 1;
        } else if (aEvolution[i].trend !== 'improving' && iCurrentStart !== null) {
          aPeriods.push({
            startIndex: iCurrentStart,
            endIndex: i - 1,
            duration: i - 1 - iCurrentStart + 1,
            improvement: aEvolution[i - 1].acceptanceRate - aEvolution[iCurrentStart].acceptanceRate
          });
          iCurrentStart = null;
        }
      }
            
      return aPeriods;
    },
        
    onCloseLearningAnalytics: function () {
      if (this._oLearningAnalyticsDialog) {
        this._oLearningAnalyticsDialog.close();
      }
    },
        
    onExportLearningData: function () {
      const oLearningData = this.getModel('view').getProperty('/learningAnalytics');
            
      // Create comprehensive learning report
      const sReportContent = this._generateLearningReport(oLearningData);
            
      // Download as CSV
      const oBlob = new Blob([sReportContent], { type: 'text/csv;charset=utf-8;' });
      const sFileName = `ai_learning_analytics_${  new Date().toISOString().split('T')[0]  }.csv`;
            
      this._downloadBlob(oBlob, sFileName);
      MessageToast.show('Learning analytics exported successfully');
    },
        
    _generateLearningReport: function (oData) {
      const aLines = [];
            
      // Header
      aLines.push('AI Learning Analytics Report');
      aLines.push(`Generated: ${  oData.lastUpdated.toLocaleString()}`);
      aLines.push('');
            
      // Overview metrics
      aLines.push('OVERVIEW METRICS');
      aLines.push(`Total Interactions,${  oData.totalInteractions}`);
      aLines.push(`Acceptance Rate,${  oData.acceptanceRate  }%`);
      aLines.push(`Rejection Rate,${  oData.rejectionRate  }%`);
      aLines.push(`Applied Suggestions,${  oData.appliedSuggestions}`);
      aLines.push(`System Confidence,${  oData.confidence  }%`);
      aLines.push(`Learning Score,${  oData.learningScore  }%`);
      aLines.push('');
            
      // Rejection analysis
      aLines.push('REJECTION ANALYSIS');
      for (const sReason in oData.rejectionAnalysis.reasonBreakdown) {
        aLines.push(`${sReason  },${  oData.rejectionAnalysis.reasonBreakdown[sReason]}`);
      }
      aLines.push('');
            
      // Field breakdown
      aLines.push('FIELD ANALYSIS');
      for (var sField in oData.rejectionAnalysis.fieldBreakdown) {
        aLines.push(`${sField  } (Rejections),${  oData.rejectionAnalysis.fieldBreakdown[sField]}`);
      }
      for (var sField in oData.acceptanceAnalysis.fieldBreakdown) {
        aLines.push(`${sField  } (Acceptances),${  oData.acceptanceAnalysis.fieldBreakdown[sField]}`);
      }
      aLines.push('');
            
      // Improvement recommendations
      aLines.push('IMPROVEMENT RECOMMENDATIONS');
      oData.improvements.forEach((oImprovement) => {
        aLines.push(`${oImprovement.title  },${  oImprovement.description  },${  oImprovement.priority}`);
      });
            
      return aLines.join('\n');
    },
        
    onResetLearningData: function () {
      MessageBox.confirm(
        'This will clear all learning data and start fresh. This action cannot be undone.',
        {
          title: 'Reset Learning Data',
          onClose: function (oAction) {
            if (oAction === MessageBox.Action.OK) {
              this._resetAllLearningData();
              MessageToast.show('Learning data has been reset');
                            
              if (this._oLearningAnalyticsDialog) {
                this._oLearningAnalyticsDialog.close();
              }
            }
          }.bind(this)
        }
      );
    },
        
    _resetAllLearningData: function () {
      this.getModel('view').setProperty('/suggestions/feedback', {
        accepted: [],
        rejected: [],
        applied: []
      });
            
      this.getModel('view').setProperty('/suggestions/rejectionPatterns', {});
      this.getModel('view').setProperty('/suggestions/temporarySuppression', {});
            
      // Reset user preferences to defaults
      this.getModel('view').setProperty('/suggestions/userPreferences', {
        enableAI: true,
        showValidationSuggestions: true,
        showCompletionSuggestions: true,
        showOptimizationSuggestions: true
      });
            
      // Reset timing preferences
      this.getModel('view').setProperty('/suggestions/showDelay', 1000);
      this.getModel('view').setProperty('/suggestions/maxSuggestions', 5);
    },
        
    _calculateEditModeTime: function () {
      if (this._editModeStartTime) {
        return Date.now() - this._editModeStartTime;
      }
      return 0;
    },

    // ===========================
    // Insights System Methods
    // ===========================
        
    _initializeInsights: function () {
      // Initialize AI-powered insights system
      this._insightsEngine = {
        analysisTimer: null,
        lastAnalysis: null,
        insightCache: {},
        activeAnalyses: [],
        historicalPatterns: {},
        comparativeData: null
      };
            
      // Setup auto-refresh if enabled
      if (this.getModel('view').getProperty('/insights/autoRefresh')) {
        this._startInsightsAutoRefresh();
      }
            
      // Perform initial analysis
      this._performInitialInsightAnalysis();
    },
        
    _performInitialInsightAnalysis: function () {
      // Delay initial analysis to allow project data to load
      setTimeout(() => {
        this._generateInsights();
      }, 2000);
    },
        
    _startInsightsAutoRefresh: function () {
      const iInterval = this.getModel('view').getProperty('/insights/refreshInterval');
            
      this._insightsEngine.analysisTimer = setInterval(() => {
        if (this.getModel('view').getProperty('/insights/enabled') && 
                    this.getModel('view').getProperty('/insights/analysisConfig/enableRealTimeAnalysis')) {
          this._generateInsights();
        }
      }, iInterval);
    },
        
    _generateInsights: function () {
      const oProjectData = this.getModel('project').getData();
      if (!oProjectData || !oProjectData.projectId) {
        return;
      }
            
      const aInsights = [];
            
      // Generate different categories of insights
      aInsights = aInsights.concat(this._generateRiskInsights(oProjectData));
      aInsights = aInsights.concat(this._generateOptimizationInsights(oProjectData));
      aInsights = aInsights.concat(this._generateComplianceInsights(oProjectData));
      aInsights = aInsights.concat(this._generatePredictionInsights(oProjectData));
      aInsights = aInsights.concat(this._generateRecommendationInsights(oProjectData));
            
      // Filter by confidence threshold
      const fConfidenceThreshold = this.getModel('view').getProperty('/insights/analysisConfig/confidenceThreshold');
      aInsights = aInsights.filter((oInsight) => {
        return oInsight.confidence >= fConfidenceThreshold;
      });
            
      // Prioritize and limit insights
      aInsights = this._prioritizeInsights(aInsights);
            
      // Update insights model
      this.getModel('view').setProperty('/insights/currentInsights', aInsights);
      this._insightsEngine.lastAnalysis = new Date();
    },
        
    _generateRiskInsights: function (oProjectData) {
      const aInsights = [];
            
      if (!this.getModel('view').getProperty('/insights/insightCategories/risk/enabled')) {
        return aInsights;
      }
            
      // Budget risk analysis
      if (oProjectData.budget && oProjectData.estimatedCost) {
        const fBudgetUtilization = oProjectData.estimatedCost / oProjectData.budget;
        if (fBudgetUtilization > 0.9) {
          aInsights.push({
            category: 'risk',
            type: 'budget_overrun',
            title: 'High Budget Risk Detected',
            description: `Project cost estimation (${  this._formatCurrency(oProjectData.estimatedCost)  
            }) is approaching budget limit (${  this._formatCurrency(oProjectData.budget)  })`,
            severity: 'critical',
            confidence: 0.95,
            impact: 'high',
            recommendations: [
              'Review cost estimates with project stakeholders',
              'Consider scope reduction or budget increase',
              'Implement stricter cost monitoring'
            ],
            icon: 'sap-icon://warning',
            trend: 'negative',
            metrics: {
              budgetUtilization: Math.round(fBudgetUtilization * 100),
              overrunRisk: 'high'
            }
          });
        }
      }
            
      // Timeline risk analysis
      if (oProjectData.startDate && oProjectData.endDate) {
        const dStart = new Date(oProjectData.startDate);
        const dEnd = new Date(oProjectData.endDate);
        const iDuration = Math.ceil((dEnd - dStart) / (1000 * 60 * 60 * 24));
                
        if (iDuration < 30) {
          aInsights.push({
            category: 'risk',
            type: 'timeline_aggressive',
            title: 'Aggressive Timeline Detected',
            description: `Project duration of ${  iDuration  } days may be insufficient for scope complexity`,
            severity: 'warning', 
            confidence: 0.8,
            impact: 'medium',
            recommendations: [
              'Validate timeline with development team',
              'Consider phased delivery approach',
              'Add buffer time for unexpected challenges'
            ],
            icon: 'sap-icon://date-time',
            trend: 'negative',
            metrics: {
              duration: iDuration,
              complexity: 'high'
            }
          });
        }
      }
            
      // Resource risk analysis
      if (oProjectData.members && oProjectData.members.length < 3) {
        aInsights.push({
          category: 'risk',
          type: 'resource_shortage',
          title: 'Limited Team Size',
          description: `Small team size (${  oProjectData.members.length  } members) may impact delivery capacity`,
          severity: 'info',
          confidence: 0.75,
          impact: 'medium',
          recommendations: [
            'Evaluate if additional resources are needed',
            'Consider cross-training team members',
            'Plan for knowledge sharing and documentation'
          ],
          icon: 'sap-icon://group',
          trend: 'neutral',
          metrics: {
            teamSize: oProjectData.members.length,
            recommendedSize: '3-8 members'
          }
        });
      }
            
      return aInsights;
    },
        
    _generateOptimizationInsights: function (oProjectData) {
      const aInsights = [];
            
      if (!this.getModel('view').getProperty('/insights/insightCategories/optimization/enabled')) {
        return aInsights;
      }
            
      // Cost optimization
      if (oProjectData.budget > 100000) {
        aInsights.push({
          category: 'optimization',
          type: 'cost_optimization',
          title: 'Cost Optimization Opportunity',
          description: 'Large budget projects can benefit from phased investment and milestone-based funding',
          severity: 'info',
          confidence: 0.8,
          impact: 'high',
          recommendations: [
            'Implement milestone-based budget allocation',
            'Consider cloud infrastructure for cost flexibility',
            'Evaluate vendor alternatives for better pricing'
          ],
          icon: 'sap-icon://money-bills',
          trend: 'positive',
          metrics: {
            potentialSavings: '15-25%',
            roi: 'high'
          }
        });
      }
            
      // Performance optimization  
      if (oProjectData.agents && oProjectData.agents.length > 0) {
        const iActiveAgents = oProjectData.agents.filter((agent) => {
          return agent.status === 'DEPLOYED';
        }).length;
                
        if (iActiveAgents > 5) {
          aInsights.push({
            category: 'optimization',
            type: 'performance_optimization',
            title: 'Agent Performance Optimization',
            description: 'Multiple active agents detected - optimize for better resource utilization',
            severity: 'info',
            confidence: 0.85,
            impact: 'medium',
            recommendations: [
              'Implement agent load balancing',
              'Consider agent consolidation opportunities',
              'Monitor and optimize agent performance metrics'
            ],
            icon: 'sap-icon://performance',
            trend: 'neutral',
            metrics: {
              activeAgents: iActiveAgents,
              optimization: 'needed'
            }
          });
        }
      }
            
      return aInsights;
    },
        
    _generateComplianceInsights: function (oProjectData) {
      const aInsights = [];
            
      if (!this.getModel('view').getProperty('/insights/insightCategories/compliance/enabled')) {
        return aInsights;
      }
            
      // Data compliance
      if (oProjectData.dataProcessing && oProjectData.dataProcessing.includes('personal')) {
        aInsights.push({
          category: 'compliance',
          type: 'data_privacy',
          title: 'Data Privacy Compliance Required',
          description: 'Project involves personal data processing - ensure GDPR/privacy compliance',
          severity: 'warning',
          confidence: 0.9,
          impact: 'critical',
          recommendations: [
            'Conduct privacy impact assessment',
            'Implement data protection measures',
            'Document data processing activities',
            'Establish user consent mechanisms'
          ],
          icon: 'sap-icon://shield',
          trend: 'neutral',
          metrics: {
            complianceRisk: 'high',
            requiresReview: true
          }
        });
      }
            
      // Security compliance
      if (oProjectData.securityLevel === undefined || oProjectData.securityLevel < 3) {
        aInsights.push({
          category: 'compliance',
          type: 'security_standards',
          title: 'Security Standards Review Needed',
          description: 'Project security level may not meet organizational standards',
          severity: 'warning',
          confidence: 0.8,
          impact: 'high',
          recommendations: [
            'Review security requirements with security team',
            'Implement appropriate security controls',
            'Conduct security risk assessment',
            'Plan security testing and validation'
          ],
          icon: 'sap-icon://locked',
          trend: 'negative',
          metrics: {
            currentLevel: oProjectData.securityLevel || 0,
            recommendedLevel: 3
          }
        });
      }
            
      return aInsights;
    },
        
    _generatePredictionInsights: function (oProjectData) {
      const aInsights = [];
            
      if (!this.getModel('view').getProperty('/insights/insightCategories/prediction/enabled')) {
        return aInsights;
      }
            
      // Delivery prediction
      if (oProjectData.startDate && oProjectData.endDate) {
        const dStart = new Date(oProjectData.startDate);
        const dEnd = new Date(oProjectData.endDate);
        const dToday = new Date();
                
        if (dStart <= dToday && dEnd >= dToday) {
          // Project is in progress
          const fProgress = (dToday - dStart) / (dEnd - dStart);
          const fActualProgress = oProjectData.progress / 100;
          const fVariance = fActualProgress - fProgress;
                    
          if (fVariance < -0.1) {
            aInsights.push({
              category: 'prediction',
              type: 'delivery_delay',
              title: 'Delivery Delay Risk Predicted',
              description: `Current progress is behind schedule - predicted delay of ${  
                Math.ceil(Math.abs(fVariance) * ((dEnd - dStart) / (1000 * 60 * 60 * 24)))  } days`,
              severity: 'warning',
              confidence: 0.85,
              impact: 'high',
              recommendations: [
                'Reassess project timeline and milestones',
                'Consider increasing team capacity',
                'Identify and remove project blockers',
                'Communicate timeline changes to stakeholders'
              ],
              icon: 'sap-icon://prediction',
              trend: 'negative',
              metrics: {
                scheduleVariance: Math.round(fVariance * 100),
                predictedDelay: Math.ceil(Math.abs(fVariance) * ((dEnd - dStart) / (1000 * 60 * 60 * 24)))
              }
            });
          }
        }
      }
            
      return aInsights;
    },
        
    _generateRecommendationInsights: function (oProjectData) {
      const aInsights = [];
            
      if (!this.getModel('view').getProperty('/insights/insightCategories/recommendation/enabled')) {
        return aInsights;
      }
            
      // Best practices recommendations
      if (!oProjectData.description || oProjectData.description.length < 50) {
        aInsights.push({
          category: 'recommendation',
          type: 'documentation_improvement',
          title: 'Improve Project Documentation',
          description: 'Detailed project description helps stakeholders understand scope and objectives',
          severity: 'info',
          confidence: 0.9,
          impact: 'medium',
          recommendations: [
            'Add comprehensive project description',
            'Document key objectives and success criteria',
            'Include technical requirements and constraints',
            'Define stakeholder roles and responsibilities'
          ],
          icon: 'sap-icon://document-text',
          trend: 'positive',
          metrics: {
            currentLength: oProjectData.description ? oProjectData.description.length : 0,
            recommendedLength: '50+ characters'
          }
        });
      }
            
      return aInsights;
    },
        
    _prioritizeInsights: function (aInsights) {
      // Sort by severity, confidence, and category priority
      return aInsights.sort((a, b) => {
        const mSeverityWeight = {
          'critical': 4,
          'warning': 3,
          'info': 2
        };
                
        const iSeverityA = mSeverityWeight[a.severity] || 1;
        const iSeverityB = mSeverityWeight[b.severity] || 1;
                
        if (iSeverityA !== iSeverityB) {
          return iSeverityB - iSeverityA; // Higher severity first
        }
                
        if (a.confidence !== b.confidence) {
          return b.confidence - a.confidence; // Higher confidence first
        }
                
        const iCategoryA = this.getModel('view').getProperty(`/insights/insightCategories/${  a.category  }/priority`) || 10;
        const iCategoryB = this.getModel('view').getProperty(`/insights/insightCategories/${  b.category  }/priority`) || 10;
                
        return iCategoryA - iCategoryB; // Lower priority number first
      });
    },
        
    _formatCurrency: function (amount) {
      return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: 'USD',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
      }).format(amount);
    },
        
    onShowInsightsPanel: function () {
      this._openInsightsPanel();
    },
        
    onConfigureInsights: function () {
      this._showInsightsConfigDialog();
    },
        
    _openInsightsPanel: function () {
      if (!this._insightsDialog) {
        this._insightsDialog = sap.ui.xmlfragment('a2a.portal.view.fragments.InsightsPanel', this);
        this.getView().addDependent(this._insightsDialog);
      }
            
      // Refresh insights before showing
      this._generateInsights();
      this._insightsDialog.open();
    },
        
    _showInsightsConfigDialog: function () {
      if (!this._insightsConfigDialog) {
        this._insightsConfigDialog = sap.ui.xmlfragment('a2a.portal.view.fragments.InsightsConfigDialog', this);
        this.getView().addDependent(this._insightsConfigDialog);
      }
      this._insightsConfigDialog.open();
    },
        
    onRefreshInsights: function () {
      this._generateInsights();
      MessageToast.show('Insights refreshed successfully');
    },
        
    onExportInsights: function () {
      const aInsights = this.getModel('view').getProperty('/insights/currentInsights');
      if (!aInsights || aInsights.length === 0) {
        MessageToast.show('No insights available to export');
        return;
      }
            
      this._exportInsightsToCSV(aInsights);
    },
        
    onCloseInsightsPanel: function () {
      if (this._insightsDialog) {
        this._insightsDialog.close();
      }
    },
        
    onResetInsightsConfig: function () {
      // Reset to default configuration
      const oDefaults = {
        enabled: true,
        autoRefresh: true,
        refreshInterval: 30000,
        showInSidebar: false,
        showInModal: true,
        insightCategories: {
          risk: { enabled: true, priority: 1 },
          optimization: { enabled: true, priority: 2 },
          compliance: { enabled: true, priority: 3 },
          prediction: { enabled: true, priority: 4 },
          recommendation: { enabled: true, priority: 5 }
        },
        analysisConfig: {
          enableRealTimeAnalysis: true,
          includeHistoricalData: true,
          enableComparativeAnalysis: true,
          enablePredictiveAnalysis: true,
          confidenceThreshold: 0.7
        },
        displayConfig: {
          maxInsightsPerCategory: 3,
          showConfidenceScores: true,
          enableDetailedView: true,
          showActionableRecommendations: true,
          groupByPriority: true
        }
      };
            
      this.getModel('view').setProperty('/insights', Object.assign(
        this.getModel('view').getProperty('/insights'), 
        oDefaults
      ));
            
      MessageToast.show('Insights configuration reset to defaults');
    },
        
    onSaveInsightsConfig: function () {
      // Save configuration to user preferences
      const oConfig = this.getModel('view').getProperty('/insights');
            
      // Update refresh interval if changed
      if (this._insightsEngine.analysisTimer) {
        clearInterval(this._insightsEngine.analysisTimer);
        this._startInsightsAutoRefresh();
      }
            
      // In production environment, save to user profile
      $.ajax({
        url: '/api/user/preferences/insights',
        method: 'POST',
        data: JSON.stringify(oConfig),
        contentType: 'application/json',
        success: function () {
          MessageToast.show('Insights configuration saved');
        }.bind(this),
        error: function () {
          MessageToast.show('Configuration saved locally');
        }
      });
            
      this._insightsConfigDialog.close();
    },
        
    onCancelInsightsConfig: function () {
      this._insightsConfigDialog.close();
    },
        
    _exportInsightsToCSV: function (aInsights) {
      const aCSVData = [];
      const aHeaders = ['Category', 'Type', 'Title', 'Severity', 'Confidence', 'Impact', 'Description', 'Recommendations'];
      aCSVData.push(aHeaders.join(','));
            
      aInsights.forEach((oInsight) => {
        const aRow = [
          oInsight.category,
          oInsight.type,
          `"${  (oInsight.title || '').replace(/"/g, '""')  }"`,
          oInsight.severity,
          `${Math.round(oInsight.confidence * 100)  }%`,
          oInsight.impact,
          `"${  (oInsight.description || '').replace(/"/g, '""')  }"`,
          `"${  (oInsight.recommendations ? oInsight.recommendations.join('; ') : '').replace(/"/g, '""')  }"`
        ];
        aCSVData.push(aRow.join(','));
      });
            
      const sCSVContent = aCSVData.join('\n');
      const oBlob = new Blob([sCSVContent], { type: 'text/csv;charset=utf-8;' });
            
      // Create download link
      const oLink = document.createElement('a');
      const sURL = URL.createObjectURL(oBlob);
      oLink.setAttribute('href', sURL);
      oLink.setAttribute('download', `project-insights-${  new Date().toISOString().split('T')[0]  }.csv`);
      oLink.style.visibility = 'hidden';
      document.body.appendChild(oLink);
      oLink.click();
      document.body.removeChild(oLink);
            
      MessageToast.show('Insights exported to CSV file');
    },
        
    onCloseSuggestions: function () {
      this._closeSuggestionsPopover();
    },
        
    _closeSuggestionsPopover: function () {
      if (this._suggestionsPopover) {
        this._suggestionsPopover.close();
      }
      this.getModel('view').setProperty('/suggestions/currentSuggestions', []);
    },
        
    _recordSuggestionFeedback: function (oSuggestion, sAction) {
      const oFeedback = this.getModel('view').getProperty('/suggestions/feedback');
      const aFeedbackArray = oFeedback[sAction] || [];
            
      aFeedbackArray.push({
        suggestion: oSuggestion,
        timestamp: new Date(),
        fieldName: this.getModel('view').getProperty('/suggestions/activeSuggestion')
      });
            
      oFeedback[sAction] = aFeedbackArray;
      this.getModel('view').setProperty('/suggestions/feedback', oFeedback);
            
      // Send feedback to backend for learning (in production)
      this._sendFeedbackToLearningService(oSuggestion, sAction);
    },
        
    _sendFeedbackToLearningService: function (oSuggestion, sAction) {
      // In production, this would send feedback to AI learning service
      $.ajax({
        url: '/api/suggestions/feedback',
        method: 'POST',
        data: JSON.stringify({
          suggestion: oSuggestion,
          action: sAction,
          userId: this.getModel('user').getProperty('/id'),
          projectId: this._projectId,
          timestamp: new Date().toISOString()
        }),
        contentType: 'application/json',
        success: function (data) {
          // eslint-disable-next-line no-console
          // eslint-disable-next-line no-console
          console.log('Feedback recorded successfully', data);
        },
        error: function (xhr, status, error) {
          console.warn('Failed to record feedback', error);
        }
      });
    },
        
    _calculateProjectMonths: function (sStartDate, sEndDate) {
      const dStart = new Date(sStartDate);
      const dEnd = new Date(sEndDate);
      return Math.round((dEnd - dStart) / (1000 * 60 * 60 * 24 * 30.44));
    },
        
    onToggleSuggestions: function () {
      const bEnabled = this.getModel('view').getProperty('/suggestions/enabled');
      this.getModel('view').setProperty('/suggestions/enabled', !bEnabled);
            
      if (!bEnabled) {
        MessageToast.show('AI Suggestions enabled - Start typing to see suggestions');
      } else {
        MessageToast.show('AI Suggestions disabled');
        this._closeSuggestionsPopover();
      }
    },
        
    onConfigureSuggestions: function () {
      this._showSuggestionsConfigDialog();
    },
        
    _showSuggestionsConfigDialog: function () {
      if (!this._suggestionsConfigDialog) {
        this._suggestionsConfigDialog = sap.ui.xmlfragment('a2a.portal.view.fragments.SuggestionsConfigDialog', this);
        this.getView().addDependent(this._suggestionsConfigDialog);
      }
      this._suggestionsConfigDialog.open();
    },
        
    onSuggestionsSettingChange: function () {
      // React to settings changes immediately
      const bEnabled = this.getModel('view').getProperty('/suggestions/userPreferences/enableAI');
      this.getModel('view').setProperty('/suggestions/enabled', bEnabled);
    },
        
    onResetSuggestionsConfig: function () {
      // Reset to default configuration
      const oDefaults = {
        enabled: true,
        autoShow: true,
        showDelay: 1000,
        maxSuggestions: 5,
        contextAware: true,
        learningEnabled: true,
        userPreferences: {
          enableAI: true,
          showValidationSuggestions: true,
          showCompletionSuggestions: true,
          showOptimizationSuggestions: true
        }
      };
            
      this.getModel('view').setProperty('/suggestions', Object.assign(
        this.getModel('view').getProperty('/suggestions'), 
        oDefaults
      ));
            
      MessageToast.show('Configuration reset to defaults');
    },
        
    onSaveSuggestionsConfig: function () {
      // Save configuration to user preferences (in production, this would persist to backend)
      const oConfig = this.getModel('view').getProperty('/suggestions');
            
      // In production environment, save to user profile
      $.ajax({
        url: '/api/user/preferences/suggestions',
        method: 'POST',
        data: JSON.stringify(oConfig),
        contentType: 'application/json',
        success: function () {
          MessageToast.show('Suggestions configuration saved');
        }.bind(this),
        error: function () {
          MessageToast.show('Configuration saved locally');
        }
      });
            
      this._suggestionsConfigDialog.close();
    },
        
    onCancelSuggestionsConfig: function () {
      this._suggestionsConfigDialog.close();
    },
        
    onPreviewSuggestion: function (oEvent) {
      const oSuggestion = oEvent.getSource().getBindingContext('view').getObject();
      const sFieldName = this._getCurrentSuggestionField();
            
      this._showSuggestionPreview(oSuggestion, sFieldName);
    },
        
    _showSuggestionPreview: function (oSuggestion, sFieldName) {
      const oControl = this._getFieldControl(sFieldName);
      if (!oControl) {
        return;
      }
            
      const sCurrentValue = oControl.getValue();
      const sPreviewValue = this._calculatePreviewValue(oSuggestion, sCurrentValue);
            
      // Show preview dialog
      const sDialogContent = `Current value: "${  sCurrentValue || '(empty)'  }"\n\n` +
                               `After applying suggestion: "${  sPreviewValue  }"\n\n` +
                               `Action: ${  oSuggestion.action.toUpperCase()}`;
            
      MessageBox.show(sDialogContent, {
        title: 'Suggestion Preview',
        actions: [
          new sap.m.Button({
            text: 'Apply Now',
            type: 'Accept',
            press: function () {
              this._processSuggestionAcceptance(oSuggestion, sFieldName);
              MessageBox.close();
            }.bind(this)
          }),
          new sap.m.Button({
            text: 'Cancel',
            type: 'Default',
            press: function () {
              MessageBox.close();
            }
          })
        ],
        emphasizedAction: 'Apply Now'
      });
    },
        
    _calculatePreviewValue: function (oSuggestion, sCurrentValue) {
      switch (oSuggestion.action) {
      case 'replace':
        return oSuggestion.value || '';
      case 'append':
        return (sCurrentValue || '') + (oSuggestion.value || '');
      case 'insert':
        const iPos = oSuggestion.insertPosition || (sCurrentValue ? sCurrentValue.length : 0);
        return (sCurrentValue || '').substring(0, iPos) + 
                           (oSuggestion.value || '') + 
                           (sCurrentValue || '').substring(iPos);
      case 'modify':
        if (oSuggestion.modifyPattern) {
          return (sCurrentValue || '').replace(
            new RegExp(oSuggestion.modifyPattern, 'g'), 
            oSuggestion.value || ''
          );
        }
        return sCurrentValue;
      default:
        return sCurrentValue;
      }
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