sap.ui.define([
  'sap/ui/core/mvc/Controller',
  'sap/ui/model/json/JSONModel',
  'sap/m/MessageToast',
  'sap/m/MessageBox',
  'sap/ui/core/format/DateFormat'
], (Controller, JSONModel, MessageToast, MessageBox, DateFormat) => {
  'use strict';
  /* global localStorage, performance */

  return Controller.extend('a2a.portal.controller.Projects', {

    onInit: function () {
      // Initialize view model
      const oViewModel = new JSONModel({
        viewMode: 'tiles',
        projects: [],
        busy: false,
        showRowActions: false,
        editMode: null, // Project ID currently being edited
        originalValues: {} // Store original values for cancel
      });
      this.getView().setModel(oViewModel, 'view');

      // Load projects data
      this._loadProjects();
            
      // Initialize hover actions
      this._initializeHoverActions();
            
      // Initialize enhanced tooltips
      this._initializeTooltips();
            
      // Initialize keyboard shortcuts for quick open
      this._initializeQuickOpenShortcuts();
            
      // Initialize action responsiveness monitoring
      this._initializeResponseMonitoring();
            
      // Load saved column configuration
      this._loadColumnConfiguration();
            
      // Load saved density settings
      this._loadDensitySettings();
            
      // Initialize saved views functionality
      this._initializeSavedViews();
    },
        
    _initializeHoverActions: function () {
      // Add mouse enter/leave handlers after view is rendered
      this.getView().addEventDelegate({
        onAfterRendering: function () {
          this._attachHoverHandlers();
        }.bind(this)
      });
    },
        
    _attachHoverHandlers: function () {
      const oTable = this.byId('projectsTable');
      if (!oTable) {
        return;
      }
            
      // Use event delegation for better performance
      jQuery(oTable.getDomRef()).on('mouseenter', '.a2a-project-row', (oEvent) => {
        this._onProjectRowHover(oEvent, true);
      });
            
      jQuery(oTable.getDomRef()).on('mouseleave', '.a2a-project-row', (oEvent) => {
        this._onProjectRowHover(oEvent, false);
      });
    },
        
    _onProjectRowHover: function (oEvent, bHover) {
      const $row = jQuery(oEvent.currentTarget);
      const oListItem = sap.ui.getCore().byId($row.attr('id'));
            
      if (oListItem) {
        const _oBindingContext = oListItem.getBindingContext('view');
                
        if (bHover) {
          // Show quick actions on hover
          $row.addClass('a2a-project-row-hover');
          this._showQuickActions(oListItem);
        } else {
          // Hide quick actions when not hovering
          $row.removeClass('a2a-project-row-hover');
          this._hideQuickActions(oListItem);
        }
      }
    },
        
    _showQuickActions: function (oListItem) {
      // Show quick action overlay
      const $quickActions = oListItem.$().find('.a2a-quick-actions');
      $quickActions.show().addClass('a2a-fade-in');
            
      // Optional: Show row actions
      const oViewModel = this.getView().getModel('view');
      oViewModel.setProperty('/showRowActions', true);
    },
        
    _hideQuickActions: function (oListItem) {
      // Hide quick action overlay
      const $quickActions = oListItem.$().find('.a2a-quick-actions');
      $quickActions.removeClass('a2a-fade-in').hide();
    },
        
    _initializeTooltips: function () {
      // Add dynamic tooltip content based on context
      this.getView().addEventDelegate({
        onAfterRendering: function () {
          this._enhanceTooltips();
        }.bind(this)
      });
    },
        
    _enhanceTooltips: function () {
      // Add dynamic tooltip content for context-sensitive help
      this._updateFilterButtonTooltip();
      this._updateExportButtonTooltip();
      this._updateDeleteButtonTooltip();
            
      // Add tooltip delay and positioning
      this._configureTooltipBehavior();
    },
        
    _updateFilterButtonTooltip: function () {
      const oFilterButton = this.byId('filterButton');
      if (oFilterButton) {
        const bHasActiveFilters = oFilterButton.hasStyleClass('hasActiveFilters');
        const sTooltip = bHasActiveFilters ? 
          this.getResourceBundle().getText('filterActiveTooltip') :
          this.getResourceBundle().getText('filterProjectsTooltip');
        oFilterButton.setTooltip(sTooltip);
      }
    },
        
    _updateExportButtonTooltip: function () {
      const oTable = this.byId('projectsTable');
      if (oTable) {
        const iSelectedCount = oTable.getSelectedIndices ? oTable.getSelectedIndices().length : 0;
        const oExportButton = oTable.getHeaderToolbar().getContent().find((oControl) => {
          return oControl.getIcon && oControl.getIcon() === 'sap-icon://download';
        });
                
        if (oExportButton) {
          const sTooltip = iSelectedCount > 0 ? 
            this.getResourceBundle().getText('exportSelectedProjectsTooltip', [iSelectedCount]) :
            this.getResourceBundle().getText('exportAllProjectsTooltip');
          oExportButton.setTooltip(sTooltip);
        }
      }
    },
        
    _updateDeleteButtonTooltip: function () {
      const oTable = this.byId('projectsTable');
      if (oTable) {
        const iSelectedCount = oTable.getSelectedIndices ? oTable.getSelectedIndices().length : 0;
        const oDeleteButton = oTable.getHeaderToolbar().getContent().find((oControl) => {
          return oControl.getIcon && oControl.getIcon() === 'sap-icon://delete';
        });
                
        if (oDeleteButton) {
          const sTooltip = iSelectedCount > 0 ? 
            this.getResourceBundle().getText('deleteSelectedProjectsTooltip', [iSelectedCount]) :
            this.getResourceBundle().getText('selectProjectsToDeleteTooltip');
          oDeleteButton.setTooltip(sTooltip);
        }
      }
    },
        
    _configureTooltipBehavior: function () {
      // Configure tooltip appearance delay and positioning
      const oView = this.getView();
      oView.$().find('[title]').each(function() {
        jQuery(this).tooltip({
          show: { delay: 500 },
          hide: { delay: 100 },
          position: {
            my: 'center bottom-10',
            at: 'center top'
          }
        });
      });
    },

    _loadProjects: function () {
      const oViewModel = this.getView().getModel('view');
      oViewModel.setProperty('/busy', true);

      jQuery.ajax({
        url: '/api/projects',
        method: 'GET',
        success: function (data) {
          oViewModel.setProperty('/projects', data.projects || []);
          oViewModel.setProperty('/busy', false);
        }.bind(this),
        error: function (xhr, status, error) {
          MessageToast.show(`Failed to load projects: ${  error}`);
          oViewModel.setProperty('/busy', false);
        }.bind(this)
      });
    },

    onCreateProject: function () {
      if (!this._oCreateDialog) {
        this._oCreateDialog = sap.ui.xmlfragment('a2a.portal.fragment.CreateProjectDialog', this);
        this.getView().addDependent(this._oCreateDialog);
      }
      this._oCreateDialog.open();
    },

    onCreateProjectConfirm: function (oEvent) {
      const oDialog = oEvent.getSource().getParent();
      const sName = sap.ui.getCore().byId('createProjectName').getValue();
      const sDescription = sap.ui.getCore().byId('createProjectDescription').getValue();

      if (!sName.trim()) {
        MessageToast.show('Please enter a project name');
        return;
      }

      const oProjectData = {
        name: sName.trim(),
        description: sDescription.trim()
      };

      jQuery.ajax({
        url: '/api/projects',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(oProjectData),
        success: function (_data) {
          MessageToast.show('Project created successfully');
          this._loadProjects();
          oDialog.close();
        }.bind(this),
        error: function (xhr, _status, _error) {
          let sMessage = 'Failed to create project';
          if (xhr.responseJSON && xhr.responseJSON.detail) {
            sMessage += `: ${  xhr.responseJSON.detail}`;
          }
          MessageToast.show(sMessage);
        }.bind(this)
      });
    },

    onCreateProjectCancel: function (oEvent) {
      oEvent.getSource().getParent().close();
    },

    onImportProjectConfirm: function (oEvent) {
      const oDialog = oEvent.getSource().getParent();
      MessageToast.show('Import functionality - coming soon');
      oDialog.close();
    },

    onImportProjectCancel: function (oEvent) {
      oEvent.getSource().getParent().close();
    },

    onImportProject: function () {
      if (!this._oImportDialog) {
        this._oImportDialog = sap.ui.xmlfragment('a2a.portal.fragment.ImportProjectDialog', this);
        this.getView().addDependent(this._oImportDialog);
      }
      this._oImportDialog.open();
    },

    onRefresh: function () {
      this._loadProjects();
      MessageToast.show('Projects refreshed');
    },

    onProjectPress: function (oEvent) {
      const oContext = oEvent.getSource().getBindingContext('view');
      const sProjectId = oContext.getProperty('project_id');
            
      // For now, just show a message since we're not using routing
      MessageToast.show(`Project selected: ${  sProjectId}`);
    },

    onViewChange: function (oEvent) {
      const sSelectedKey = oEvent.getParameter('item').getKey();
      const oViewModel = this.getView().getModel('view');
      oViewModel.setProperty('/viewMode', sSelectedKey);
    },

    onSearch: function (oEvent) {
      const sQuery = oEvent.getParameter('query');
      const oTable = this.byId('projectsTable');
      const oBinding = oTable.getBinding('items');

      if (sQuery && sQuery.length > 0) {
        const oFilter = new sap.ui.model.Filter([
          new sap.ui.model.Filter('name', sap.ui.model.FilterOperator.Contains, sQuery),
          new sap.ui.model.Filter('description', sap.ui.model.FilterOperator.Contains, sQuery),
          new sap.ui.model.Filter('project_id', sap.ui.model.FilterOperator.Contains, sQuery)
        ], false);
        oBinding.filter([oFilter]);
      } else {
        oBinding.filter([]);
      }
    },

    onOpenFilterDialog: function () {
      if (!this._oFilterDialog) {
        this._oFilterDialog = sap.ui.xmlfragment('a2a.portal.fragment.FilterDialog', this);
        this.getView().addDependent(this._oFilterDialog);
      }
      this._oFilterDialog.open();
    },

    onFilterConfirm: function (oEvent) {
      const oTable = this.byId('projectsTable');
      const oBinding = oTable.getBinding('items');
      const aFilters = [];

      // Get filter settings from the event
      const oFilterSettings = oEvent.getParameter('filterSettings');
            
      if (oFilterSettings) {
        // Status filter
        if (oFilterSettings.status && oFilterSettings.status.length > 0) {
          const aStatusFilters = oFilterSettings.status.map((sStatus) => {
            return new sap.ui.model.Filter('status', sap.ui.model.FilterOperator.EQ, sStatus);
          });
          aFilters.push(new sap.ui.model.Filter(aStatusFilters, false));
        }

        // Date range filter
        if (oFilterSettings.dateFrom) {
          aFilters.push(new sap.ui.model.Filter('last_modified', sap.ui.model.FilterOperator.GE, oFilterSettings.dateFrom));
        }
        if (oFilterSettings.dateTo) {
          aFilters.push(new sap.ui.model.Filter('last_modified', sap.ui.model.FilterOperator.LE, oFilterSettings.dateTo));
        }

        // Agent count filter
        if (oFilterSettings.minAgents !== undefined) {
          aFilters.push(new sap.ui.model.Filter('agents', sap.ui.model.FilterOperator.GE, oFilterSettings.minAgents));
        }
      }

      // Apply combined filters
      oBinding.filter(aFilters);
            
      // Update filter indicator
      this._updateFilterIndicator(aFilters.length > 0);
    },

    _updateFilterIndicator: function (bHasFilters) {
      const oFilterButton = this.byId('filterButton');
      if (oFilterButton && bHasFilters) {
        oFilterButton.addStyleClass('hasActiveFilters');
      } else if (oFilterButton) {
        oFilterButton.removeStyleClass('hasActiveFilters');
      }
            
      // Update tooltip to reflect current state
      this._updateFilterButtonTooltip();
    },

    onClearFilters: function () {
      const oTable = this.byId('projectsTable');
      const oBinding = oTable.getBinding('items');
      oBinding.filter([]);
      this._updateFilterIndicator(false);
      MessageToast.show('All filters cleared');
    },
        
    onSelectionChange: function (_oEvent) {
      // Update tooltips when selection changes
      setTimeout(() => {
        this._updateExportButtonTooltip();
        this._updateDeleteButtonTooltip();
      }, 100); // Small delay to ensure selection is processed
    },
        
    // Quick action handlers
    onQuickOpenProject: function (oEvent) {
      const oBindingContext = oEvent.getSource().getBindingContext('view');
      const oProject = oBindingContext.getObject();
            
      this._performQuickOpen(oProject);
    },
        
    _performQuickOpen: function (oProject) {
      // Show loading state
      const oViewModel = this.getView().getModel('view');
      oViewModel.setProperty('/busy', true);
            
      // Validate project accessibility
      this._validateProjectAccess(oProject).then((bCanAccess) => {
        if (!bCanAccess) {
          MessageToast.show('Project access denied or project unavailable');
          oViewModel.setProperty('/busy', false);
          return;
        }
                
        // Navigate to project workspace
        this._navigateToProjectWorkspace(oProject);
                
      }).catch((error) => {
        MessageToast.show(`Failed to access project: ${  error.message}`);
        oViewModel.setProperty('/busy', false);
      });
    },
        
    _validateProjectAccess: function (oProject) {
      // Check if project is accessible and user has permissions
      return new Promise((resolve) => {
        jQuery.ajax({
          url: `/api/projects/${  oProject.project_id  }/access`,
          method: 'GET',
          success: function(data) {
            resolve(data.hasAccess === true);
          },
          error: function() {
            resolve(false);
          }
        });
      });
    },
        
    _navigateToProjectWorkspace: function (oProject) {
      const oViewModel = this.getView().getModel('view');
            
      // Record quick open action for analytics
      this._recordQuickOpenAction(oProject);
            
      // Navigate based on project type
      let sRoute;
      switch (oProject.status) {
      case 'Active':
        sRoute = 'projectWorkspace';
        break;
      case 'Draft':
        sRoute = 'projectEditor';
        break;
      case 'Archived':
        sRoute = 'projectViewer';
        break;
      default:
        sRoute = 'projectDetails';
      }
            
      // Perform navigation
      const oRouter = this.getRouter();
      if (oRouter) {
        oRouter.navTo(sRoute, {
          projectId: oProject.project_id
        });
      } else {
        // Fallback: open in new tab
        window.open(`/projects/${  oProject.project_id  }/workspace`, '_blank');
      }
            
      oViewModel.setProperty('/busy', false);
    },
        
    _recordQuickOpenAction: function (oProject) {
      // Record analytics for quick open usage
      if (window.analytics) {
        window.analytics.track('Quick Open Project', {
          projectId: oProject.project_id,
          projectName: oProject.name,
          projectStatus: oProject.status,
          timestamp: new Date().toISOString()
        });
      }
    },
        
    _initializeQuickOpenShortcuts: function () {
      // Add keyboard shortcuts for quick actions
      document.addEventListener('keydown', (oEvent) => {
        // Ctrl+Enter or Cmd+Enter for quick open selected project
        if ((oEvent.ctrlKey || oEvent.metaKey) && oEvent.key === 'Enter') {
          oEvent.preventDefault();
          this._handleQuickOpenShortcut();
        }
                
        // Ctrl+Shift+O for quick open dialog
        if ((oEvent.ctrlKey || oEvent.metaKey) && oEvent.shiftKey && oEvent.key === 'O') {
          oEvent.preventDefault();
          this._openQuickAccessDialog();
        }
      });
    },
        
    _handleQuickOpenShortcut: function () {
      const oTable = this.byId('projectsTable');
      if (!oTable) {
        return;
      }
            
      const aSelectedIndices = oTable.getSelectedIndices ? oTable.getSelectedIndices() : [];
      if (aSelectedIndices.length === 1) {
        // Quick open the selected project
        const oContext = oTable.getContextByIndex(aSelectedIndices[0]);
        if (oContext) {
          const oProject = oContext.getObject();
          this._performQuickOpen(oProject);
        }
      } else if (aSelectedIndices.length === 0) {
        MessageToast.show('Select a project to quick open (Ctrl+Enter)');
      } else {
        MessageToast.show('Select only one project for quick open');
      }
    },
        
    _openQuickAccessDialog: function () {
      // Open a quick access dialog for project search and navigation
      if (!this._oQuickAccessDialog) {
        this._oQuickAccessDialog = new sap.m.Dialog({
          title: 'Quick Open Project',
          contentWidth: '600px',
          contentHeight: '400px',
          draggable: true,
          resizable: true,
          content: [
            new sap.m.SearchField({
              id: 'quickOpenSearch',
              placeholder: 'Type project name to search...',
              width: '100%',
              liveChange: this._onQuickOpenSearch.bind(this)
            }),
            new sap.m.List({
              id: 'quickOpenList',
              mode: 'SingleSelectMaster',
              itemPress: this._onQuickOpenItemPress.bind(this)
            })
          ],
          buttons: [
            new sap.m.Button({
              text: 'Open',
              type: 'Emphasized',
              press: this._onQuickOpenConfirm.bind(this)
            }),
            new sap.m.Button({
              text: 'Cancel',
              press: this._onQuickOpenCancel.bind(this)
            })
          ]
        });
        this.getView().addDependent(this._oQuickAccessDialog);
      }
            
      this._populateQuickOpenList();
      this._oQuickAccessDialog.open();
            
      // Focus on search field
      setTimeout(() => {
        sap.ui.getCore().byId('quickOpenSearch').focus();
      }, 100);
    },
        
    _populateQuickOpenList: function () {
      const oList = sap.ui.getCore().byId('quickOpenList');
      const oModel = this.getView().getModel('view');
      const aProjects = oModel.getProperty('/projects') || [];
            
      oList.destroyItems();
            
      aProjects.forEach((oProject) => {
        oList.addItem(new sap.m.StandardListItem({
          title: oProject.name,
          description: oProject.description || 'No description',
          info: oProject.status,
          infoState: this._getProjectStatusState(oProject.status),
          icon: 'sap-icon://folder',
          data: {
            project: oProject
          }
        }));
      });
    },
        
    _getProjectStatusState: function (sStatus) {
      switch (sStatus) {
      case 'Active': return 'Success';
      case 'Draft': return 'Warning';
      case 'Archived': return 'Information';
      case 'Error': return 'Error';
      default: return 'None';
      }
    },
        
    _onQuickOpenSearch: function (oEvent) {
      const sQuery = oEvent.getParameter('newValue').toLowerCase();
      const oList = sap.ui.getCore().byId('quickOpenList');
      const aItems = oList.getItems();
            
      aItems.forEach((oItem) => {
        const sTitle = oItem.getTitle().toLowerCase();
        const sDescription = (oItem.getDescription() || '').toLowerCase();
        const bVisible = sTitle.includes(sQuery) || sDescription.includes(sQuery);
        oItem.setVisible(bVisible);
      });
    },
        
    _onQuickOpenItemPress: function (oEvent) {
      const oItem = oEvent.getParameter('listItem');
      const oProject = oItem.data('project');
      this._performQuickOpen(oProject);
      this._oQuickAccessDialog.close();
    },
        
    _onQuickOpenConfirm: function () {
      const oList = sap.ui.getCore().byId('quickOpenList');
      const oSelectedItem = oList.getSelectedItem();
            
      if (oSelectedItem) {
        const oProject = oSelectedItem.data('project');
        this._performQuickOpen(oProject);
        this._oQuickAccessDialog.close();
      } else {
        MessageToast.show('Please select a project to open');
      }
    },
        
    _onQuickOpenCancel: function () {
      this._oQuickAccessDialog.close();
    },
        
    // Quick edit functionality
    onQuickEditProject: function (oEvent) {
      const sActionId = `quickEdit_${  Date.now()}`;
      this._recordActionStart({target: oEvent.getSource().getDomRef()});
            
      const oBindingContext = oEvent.getSource().getBindingContext('view');
      const oProject = oBindingContext.getObject();
            
      try {
        this._startQuickEdit(oProject);
        this._recordActionComplete(sActionId, true);
      } catch (error) {
        this._recordActionComplete(sActionId, false);
        throw error;
      }
    },
        
    _startQuickEdit: function (oProject) {
      const oViewModel = this.getView().getModel('view');
            
      // Store original values for cancel
      oViewModel.setProperty('/originalValues', {
        name: oProject.name,
        description: oProject.description
      });
            
      // Set edit mode for this project
      oViewModel.setProperty('/editMode', oProject.project_id);
            
      // Focus on the name input field
      setTimeout(() => {
        const $nameInput = jQuery('.a2a-inline-edit-name input').filter(':visible').first();
        if ($nameInput.length) {
          $nameInput.focus().select();
        }
      }, 100);
    },
        
    onProjectNameChange: function (oEvent) {
      // Validate project name in real-time
      const sValue = oEvent.getParameter('value');
      const oInput = oEvent.getSource();
            
      if (!sValue || sValue.trim().length === 0) {
        oInput.setValueState('Error');
        oInput.setValueStateText('Project name is required');
      } else if (sValue.length > 100) {
        oInput.setValueState('Error');
        oInput.setValueStateText('Project name must be 100 characters or less');
      } else {
        oInput.setValueState('None');
        oInput.setValueStateText('');
      }
    },
        
    onProjectNameSubmit: function (_oEvent) {
      // Handle Enter key press in name field
      this._validateAndSaveQuickEdit();
    },
        
    onProjectDescriptionChange: function (oEvent) {
      // Validate description length
      const sValue = oEvent.getParameter('value');
      const oTextArea = oEvent.getSource();
            
      if (sValue && sValue.length > 500) {
        oTextArea.setValueState('Warning');
        oTextArea.setValueStateText(`Description is getting long (${  sValue.length  }/500 chars)`);
      } else {
        oTextArea.setValueState('None');
        oTextArea.setValueStateText('');
      }
    },
        
    onProjectDescriptionSubmit: function (_oEvent) {
      // Handle Enter key press in description field
      this._validateAndSaveQuickEdit();
    },
        
    onSaveQuickEdit: function (_oEvent) {
      this._validateAndSaveQuickEdit();
    },
        
    onCancelQuickEdit: function (_oEvent) {
      this._cancelQuickEdit();
    },
        
    _validateAndSaveQuickEdit: function () {
      const oViewModel = this.getView().getModel('view');
      const sEditingProjectId = oViewModel.getProperty('/editMode');
            
      if (!sEditingProjectId) {
        return;
      }
            
      // Find the project being edited
      const aProjects = oViewModel.getProperty('/projects') || [];
      const oProject = aProjects.find((p) => {
        return p.project_id === sEditingProjectId; 
      });
            
      if (!oProject) {
        return;
      }
            
      // Validate inputs
      const bValid = this._validateQuickEditInputs(oProject);
      if (!bValid) {
        return;
      }
            
      // Save changes
      this._saveProjectChanges(oProject);
    },
        
    _validateQuickEditInputs: function (oProject) {
      let bValid = true;
            
      // Validate name
      if (!oProject.name || oProject.name.trim().length === 0) {
        MessageToast.show('Project name is required');
        bValid = false;
      } else if (oProject.name.length > 100) {
        MessageToast.show('Project name must be 100 characters or less');
        bValid = false;
      }
            
      // Validate description length
      if (oProject.description && oProject.description.length > 500) {
        MessageToast.show('Description must be 500 characters or less');
        bValid = false;
      }
            
      return bValid;
    },
        
    _saveProjectChanges: function (oProject) {
      const oViewModel = this.getView().getModel('view');
      const iApiStartTime = performance.now();
            
      // Show saving state
      oViewModel.setProperty('/busy', true);
            
      // API call to save changes
      jQuery.ajax({
        url: `/api/projects/${  oProject.project_id}`,
        method: 'PATCH',
        contentType: 'application/json',
        data: JSON.stringify({
          name: oProject.name.trim(),
          description: oProject.description ? oProject.description.trim() : ''
        }),
        success: function(data) {
          const iApiEndTime = performance.now();
          const iApiDuration = iApiEndTime - iApiStartTime;
                    
          // Record API performance
          this._actionMetrics.apiCalls.push({
            endpoint: `PATCH /api/projects/${  oProject.project_id}`,
            duration: iApiDuration,
            success: true,
            timestamp: new Date().toISOString()
          });
                    
          // Update project with server response
          Object.assign(oProject, data);
          oViewModel.updateBindings();
                    
          MessageToast.show('Project updated successfully');
          this._exitQuickEdit();
                    
          // Log slow API calls
          if (iApiDuration > 1000) {
            console.warn('Slow API call detected:', `${iApiDuration  }ms`);
          }
        }.bind(this),
        error: function(xhr) {
          const iApiEndTime = performance.now();
          const iApiDuration = iApiEndTime - iApiStartTime;
                    
          // Record API failure
          this._actionMetrics.apiCalls.push({
            endpoint: `PATCH /api/projects/${  oProject.project_id}`,
            duration: iApiDuration,
            success: false,
            timestamp: new Date().toISOString()
          });
                    
          let sError = 'Failed to update project';
          if (xhr.responseJSON && xhr.responseJSON.message) {
            sError += `: ${  xhr.responseJSON.message}`;
          }
          MessageToast.show(sError);
        }.bind(this),
        complete: function() {
          oViewModel.setProperty('/busy', false);
        }.bind(this)
      });
    },
        
    _cancelQuickEdit: function () {
      const oViewModel = this.getView().getModel('view');
      const sEditingProjectId = oViewModel.getProperty('/editMode');
      const oOriginalValues = oViewModel.getProperty('/originalValues');
            
      if (!sEditingProjectId || !oOriginalValues) {
        return;
      }
            
      // Find and restore original values
      const aProjects = oViewModel.getProperty('/projects') || [];
      const oProject = aProjects.find((p) => {
        return p.project_id === sEditingProjectId; 
      });
            
      if (oProject) {
        oProject.name = oOriginalValues.name;
        oProject.description = oOriginalValues.description;
        oViewModel.updateBindings();
      }
            
      this._exitQuickEdit();
      MessageToast.show('Changes cancelled');
    },
        
    _exitQuickEdit: function () {
      const oViewModel = this.getView().getModel('view');
      oViewModel.setProperty('/editMode', null);
      oViewModel.setProperty('/originalValues', {});
    },
        
    // Action responsiveness monitoring
    _initializeResponseMonitoring: function () {
      this._actionMetrics = {
        clickResponses: [],
        hoverResponses: [],
        apiCalls: [],
        renderTimes: []
      };
            
      // Monitor all button clicks for responsiveness
      this._attachClickMonitoring();
            
      // Monitor render performance
      this._monitorRenderPerformance();
    },
        
    _attachClickMonitoring: function () {
      // Use event delegation to monitor all button clicks
      const oView = this.getView();
      oView.addEventDelegate({
        onclick: function(oEvent) {
          if (oEvent.target.tagName === 'BUTTON' || 
                        oEvent.target.closest('button') ||
                        oEvent.target.classList.contains('sapMBtn')) {
            this._recordActionStart(oEvent);
          }
        }.bind(this)
      });
    },
        
    _recordActionStart: function (oEvent) {
      const sActionId = this._getActionId(oEvent.target);
      const iStartTime = performance.now();
            
      // Store start time for this action
      this._currentAction = {
        id: sActionId,
        startTime: iStartTime,
        element: oEvent.target
      };
            
      // Provide immediate visual feedback
      this._showActionFeedback(oEvent.target);
            
      // Set timeout to detect slow responses
      setTimeout(() => {
        if (this._currentAction && this._currentAction.id === sActionId) {
          this._handleSlowAction(sActionId);
        }
      }, 300); // 300ms threshold for slow actions
    },
        
    _recordActionComplete: function (sActionId, bSuccess) {
      if (!this._currentAction || this._currentAction.id !== sActionId) {
        return;
      }
            
      const iEndTime = performance.now();
      const iDuration = iEndTime - this._currentAction.startTime;
            
      // Record metrics
      this._actionMetrics.clickResponses.push({
        action: sActionId,
        duration: iDuration,
        success: bSuccess,
        timestamp: new Date().toISOString()
      });
            
      // Log slow actions
      if (iDuration > 100) {
        console.warn('Slow action detected:', sActionId, `${iDuration  }ms`);
      }
            
      // Clear current action
      this._currentAction = null;
            
      // Hide loading feedback
      this._hideActionFeedback();
    },
        
    _getActionId: function (oElement) {
      // Determine action ID from element
      const sId = oElement.id || oElement.closest('[id]')?.id || '';
      const sClass = oElement.className || '';
      const sIcon = oElement.querySelector('.sapUiIcon')?.getAttribute('data-sap-ui-icon-content') || '';
            
      return `${sId  }_${  sClass.split(' ')[0]  }_${  sIcon}`;
    },
        
    _showActionFeedback: function (oElement) {
      // Add visual feedback for action start
      const $element = jQuery(oElement).closest('.sapMBtn');
      $element.addClass('a2a-action-loading');
            
      // Add subtle loading indicator
      if (!$element.find('.a2a-loading-indicator').length) {
        $element.append('<span class="a2a-loading-indicator"></span>');
      }
    },
        
    _hideActionFeedback: function () {
      // Remove loading indicators
      jQuery('.a2a-action-loading').removeClass('a2a-action-loading');
      jQuery('.a2a-loading-indicator').remove();
    },
        
    _handleSlowAction: function (sActionId) {
      console.warn('Action taking longer than expected:', sActionId);
            
      // Show progress indicator for slow actions
      if (this._currentAction) {
        const $element = jQuery(this._currentAction.element).closest('.sapMBtn');
        $element.addClass('a2a-slow-action');
      }
    },
        
    _monitorRenderPerformance: function () {
      // Monitor table rendering performance
      this.getView().addEventDelegate({
        onAfterRendering: function() {
          const iRenderTime = performance.now();
          this._actionMetrics.renderTimes.push({
            timestamp: new Date().toISOString(),
            renderTime: iRenderTime
          });
                    
          // Log slow renders
          if (this._lastRenderTime && (iRenderTime - this._lastRenderTime) > 100) {
            console.warn('Slow render detected:', `${iRenderTime - this._lastRenderTime  }ms`);
          }
                    
          this._lastRenderTime = iRenderTime;
        }.bind(this)
      });
    },
        
    // Performance metrics access (for debugging/development)
    getPerformanceMetrics: function () {
      if (!this._actionMetrics) {
        return null;
      }
            
      // Calculate averages and statistics
      const oMetrics = {
        clickResponses: this._calculateStats(this._actionMetrics.clickResponses, 'duration'),
        apiCalls: this._calculateStats(this._actionMetrics.apiCalls, 'duration'),
        totalActions: this._actionMetrics.clickResponses.length,
        slowActions: this._actionMetrics.clickResponses.filter((a) => {
          return a.duration > 100; 
        }),
        failedActions: this._actionMetrics.clickResponses.filter((a) => {
          return !a.success; 
        }),
        apiSuccessRate: this._calculateSuccessRate(this._actionMetrics.apiCalls)
      };
            
      return oMetrics;
    },
        
    _calculateStats: function (aData, sProperty) {
      if (!aData || aData.length === 0) {
        return { average: 0, min: 0, max: 0, count: 0 };
      }
            
      const aValues = aData.map((item) => {
        return item[sProperty]; 
      });
      const iSum = aValues.reduce((sum, val) => {
        return sum + val; 
      }, 0);
            
      return {
        average: Math.round(iSum / aValues.length * 100) / 100,
        min: Math.min(...aValues),
        max: Math.max(...aValues),
        count: aValues.length
      };
    },
        
    _calculateSuccessRate: function (aApiCalls) {
      if (!aApiCalls || aApiCalls.length === 0) {
        return 100;
      }
            
      const iSuccessCount = aApiCalls.filter((call) => {
        return call.success; 
      }).length;
      return Math.round((iSuccessCount / aApiCalls.length) * 100);
    },
        
    // Debug method to log performance summary
    logPerformanceSummary: function () {
      const oMetrics = this.getPerformanceMetrics();
      if (!oMetrics) {
        // eslint-disable-next-line no-console
        // eslint-disable-next-line no-console
        console.log('Performance monitoring not initialized');
        return;
      }
            
      // eslint-disable-next-line no-console
      console.group('Projects View Performance Summary');
      // eslint-disable-next-line no-console
      // eslint-disable-next-line no-console
      console.log('Total Actions:', oMetrics.totalActions);
      // eslint-disable-next-line no-console
      // eslint-disable-next-line no-console
      console.log('Average Action Response:', `${oMetrics.clickResponses.average  }ms`);
      // eslint-disable-next-line no-console
      // eslint-disable-next-line no-console
      console.log('Slow Actions (>100ms):', oMetrics.slowActions.length);
      // eslint-disable-next-line no-console
      // eslint-disable-next-line no-console
      console.log('Failed Actions:', oMetrics.failedActions.length);
      // eslint-disable-next-line no-console
      // eslint-disable-next-line no-console
      console.log('API Success Rate:', `${oMetrics.apiSuccessRate  }%`);
      // eslint-disable-next-line no-console
      // eslint-disable-next-line no-console
      console.log('Average API Response:', `${oMetrics.apiCalls.average  }ms`);
            
      if (oMetrics.slowActions.length > 0) {
        // eslint-disable-next-line no-console
        console.warn('Slow actions detected:', oMetrics.slowActions);
      }
            
      // eslint-disable-next-line no-console
      console.groupEnd();
    },
        
    onOpenColumnConfigDialog: function (_oEvent) {
      if (!this._oColumnConfigDialog) {
        this._oColumnConfigDialog = sap.ui.xmlfragment(
          'a2a.portal.fragment.ColumnConfigDialog', 
          this
        );
        this.getView().addDependent(this._oColumnConfigDialog);
      }
            
      // Initialize column config model
      this._initializeColumnConfigModel();
      this._oColumnConfigDialog.open();
    },
        
    _initializeColumnConfigModel: function () {
      const oTable = this.byId('projectsTable');
      const aColumns = oTable.getColumns();
            
      const aColumnConfig = aColumns.map((oColumn, iIndex) => {
        return {
          index: iIndex,
          label: oColumn.getHeader().getText(),
          visible: oColumn.getVisible(),
          key: oColumn.getId().split('--').pop() // Extract key from ID
        };
      });
            
      const oConfigModel = new JSONModel({
        columns: aColumnConfig,
        originalConfig: JSON.parse(JSON.stringify(aColumnConfig))
      });
            
      this._oColumnConfigDialog.setModel(oConfigModel, 'columnConfig');
    },
        
    onColumnVisibilityChange: function (oEvent) {
      const bVisible = oEvent.getParameter('selected');
      const oSource = oEvent.getSource();
      const oBindingContext = oSource.getBindingContext('columnConfig');
      const iIndex = oBindingContext.getProperty('index');
            
      // Update model
      oBindingContext.getModel().setProperty(
        `${oBindingContext.getPath()  }/visible`, 
        bVisible
      );
            
      // Apply change immediately for preview
      const oTable = this.byId('projectsTable');
      const aColumns = oTable.getColumns();
      if (aColumns[iIndex]) {
        aColumns[iIndex].setVisible(bVisible);
      }
    },
        
    onSaveColumnConfig: function () {
      const oConfigModel = this._oColumnConfigDialog.getModel('columnConfig');
      const aColumnConfig = oConfigModel.getProperty('/columns');
            
      // Apply all column visibility settings
      const oTable = this.byId('projectsTable');
      const aColumns = oTable.getColumns();
            
      aColumnConfig.forEach((oConfig, iIndex) => {
        if (aColumns[iIndex]) {
          aColumns[iIndex].setVisible(oConfig.visible);
        }
      });
            
      // Save to user preferences
      this._saveColumnConfiguration(aColumnConfig);
            
      MessageToast.show('Column configuration saved');
      this._oColumnConfigDialog.close();
    },
        
    onCancelColumnConfig: function () {
      // Restore original configuration
      const oConfigModel = this._oColumnConfigDialog.getModel('columnConfig');
      const aOriginalConfig = oConfigModel.getProperty('/originalConfig');
            
      const oTable = this.byId('projectsTable');
      const aColumns = oTable.getColumns();
            
      aOriginalConfig.forEach((oConfig, iIndex) => {
        if (aColumns[iIndex]) {
          aColumns[iIndex].setVisible(oConfig.visible);
        }
      });
            
      this._oColumnConfigDialog.close();
    },
        
    onResetColumnConfig: function () {
      // Reset to default visibility
      const aDefaultConfig = this._getDefaultColumnConfig();
      const oTable = this.byId('projectsTable');
      const aColumns = oTable.getColumns();
            
      aDefaultConfig.forEach((oConfig, iIndex) => {
        if (aColumns[iIndex]) {
          aColumns[iIndex].setVisible(oConfig.visible);
        }
      });
            
      // Update dialog model
      const oConfigModel = this._oColumnConfigDialog.getModel('columnConfig');
      oConfigModel.setProperty('/columns', JSON.parse(JSON.stringify(aDefaultConfig)));
            
      MessageToast.show('Column configuration reset to defaults');
    },
        
    _getDefaultColumnConfig: function () {
      return [
        { index: 0, label: 'Name', visible: true, key: 'name' },
        { index: 1, label: 'Description', visible: true, key: 'description' },
        { index: 2, label: 'Agents', visible: true, key: 'agents' },
        { index: 3, label: 'Last Modified', visible: true, key: 'lastModified' },
        { index: 4, label: 'Status', visible: true, key: 'status' },
        { index: 5, label: 'Actions', visible: true, key: 'actions' }
      ];
    },
        
    _saveColumnConfiguration: function (aColumnConfig) {
      // Save to localStorage for persistence
      const sUserId = 'current_user'; // Would come from user model in real app
      const sKey = `a2a_projects_column_config_${  sUserId}`;
            
      localStorage.setItem(sKey, JSON.stringify(aColumnConfig));
            
      // Also save to backend if available
      this._saveColumnConfigToBackend(aColumnConfig);
    },
        
    _loadColumnConfiguration: function () {
      const sUserId = 'current_user';
      const sKey = `a2a_projects_column_config_${  sUserId}`;
            
      try {
        const sStoredConfig = localStorage.getItem(sKey);
        if (sStoredConfig) {
          const aColumnConfig = JSON.parse(sStoredConfig);
          this._applyColumnConfiguration(aColumnConfig);
        }
      } catch (e) {
        console.warn('Failed to load column configuration:', e);
      }
    },
        
    _applyColumnConfiguration: function (aColumnConfig) {
      const oTable = this.byId('projectsTable');
      const aColumns = oTable.getColumns();
            
      aColumnConfig.forEach((oConfig, iIndex) => {
        if (aColumns[iIndex]) {
          aColumns[iIndex].setVisible(oConfig.visible);
        }
      });
    },
        
    _saveColumnConfigToBackend: function (aColumnConfig) {
      // API call to save configuration
      jQuery.ajax({
        url: '/api/user/preferences/column-config',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
          view: 'projects',
          configuration: aColumnConfig
        }),
        success: function () {
          // eslint-disable-next-line no-console
          // eslint-disable-next-line no-console
          console.log('Column configuration saved to backend');
        },
        error: function () {
          console.warn('Failed to save column configuration to backend');
        }
      });
    },
        
    onSetDensity: function (oEvent) {
      const oSource = oEvent.getSource();
      const sCustomData = oSource.data('key');
      const sDensityKey = sCustomData || 'cozy';
            
      this._applyDensity(sDensityKey);
      this._saveDensitySettings(sDensityKey);
            
      MessageToast.show(`Density changed to ${  sDensityKey}`);
    },
        
    _applyDensity: function (sDensityKey) {
      const oView = this.getView();
            
      // Remove existing density classes
      oView.removeStyleClass('sapUiSizeCozy');
      oView.removeStyleClass('sapUiSizeCompact'); 
      oView.removeStyleClass('sapUiSizeCondensed');
            
      // Add new density class
      let sDensityClass;
      switch (sDensityKey) {
      case 'compact':
        sDensityClass = 'sapUiSizeCompact';
        break;
      case 'condensed':
        sDensityClass = 'sapUiSizeCondensed';
        break;
      default:
        sDensityClass = 'sapUiSizeCozy';
        sDensityKey = 'cozy';
        break;
      }
            
      oView.addStyleClass(sDensityClass);
            
      // Update table row heights based on density
      this._updateTableRowHeights(sDensityKey);
            
      // Store current density for future reference
      this._sCurrentDensity = sDensityKey;
    },
        
    _updateTableRowHeights: function (sDensityKey) {
      const oTable = this.byId('projectsTable');
      if (!oTable) {
        return;
      }
            
      let iRowHeight;
      switch (sDensityKey) {
      case 'compact':
        iRowHeight = 32;
        break;
      case 'condensed':
        iRowHeight = 28;
        break;
      default: // cozy
        iRowHeight = 48;
        break;
      }
            
      // Apply row height if table supports it
      if (oTable.setRowHeight) {
        oTable.setRowHeight(iRowHeight);
      }
            
      // Add density-specific CSS class to table
      oTable.removeStyleClass('a2a-density-cozy');
      oTable.removeStyleClass('a2a-density-compact');
      oTable.removeStyleClass('a2a-density-condensed');
      oTable.addStyleClass(`a2a-density-${  sDensityKey}`);
    },
        
    _loadDensitySettings: function () {
      const sUserId = 'current_user';
      const sKey = `a2a_projects_density_${  sUserId}`;
            
      try {
        const sStoredDensity = localStorage.getItem(sKey);
        if (sStoredDensity && ['cozy', 'compact', 'condensed'].includes(sStoredDensity)) {
          this._applyDensity(sStoredDensity);
        } else {
          this._applyDensity('cozy'); // Default to cozy
        }
      } catch (e) {
        console.warn('Failed to load density settings:', e);
        this._applyDensity('cozy');
      }
    },
        
    _saveDensitySettings: function (sDensityKey) {
      // Save to localStorage for persistence
      const sUserId = 'current_user';
      const sKey = `a2a_projects_density_${  sUserId}`;
            
      try {
        localStorage.setItem(sKey, sDensityKey);
      } catch (e) {
        console.warn('Failed to save density settings:', e);
      }
            
      // Also save to backend
      this._saveDensityToBackend(sDensityKey);
    },
        
    _saveDensityToBackend: function (sDensityKey) {
      // API call to save density preference
      jQuery.ajax({
        url: '/api/user/preferences/density',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
          view: 'projects',
          density: sDensityKey
        }),
        success: function () {
          // eslint-disable-next-line no-console
          // eslint-disable-next-line no-console
          console.log('Density settings saved to backend');
        },
        error: function () {
          console.warn('Failed to save density settings to backend');
        }
      });
    },
        
    getCurrentDensity: function () {
      return this._sCurrentDensity || 'cozy';
    },
        
    _initializeSavedViews: function () {
      // Initialize saved views model
      const oSavedViewsModel = new JSONModel({
        views: [],
        currentView: null,
        defaultView: this._getDefaultViewState()
      });
      this.getView().setModel(oSavedViewsModel, 'savedViews');
            
      // Load saved views
      this._loadSavedViews();
            
      // Apply last used view or default
      this._applyLastUsedView();
    },
        
    _getDefaultViewState: function () {
      return {
        id: 'default',
        name: 'Default View',
        description: 'System default view configuration',
        isDefault: true,
        viewMode: 'tiles',
        density: 'cozy',
        columnConfig: this._getDefaultColumnConfig(),
        filters: {
          search: '',
          status: '',
          dateRange: null,
          activeFilters: []
        },
        sorting: {
          field: 'last_modified',
          descending: true
        }
      };
    },
        
    getCurrentViewState: function () {
      const _oTable = this.byId('projectsTable');
      const oSearchField = this.byId('searchField');
      const oViewModel = this.getView().getModel('view');
            
      return {
        viewMode: oViewModel.getProperty('/viewMode') || 'tiles',
        density: this.getCurrentDensity(),
        columnConfig: this._getCurrentColumnConfig(),
        filters: {
          search: oSearchField ? oSearchField.getValue() : '',
          status: this._getCurrentStatusFilter(),
          dateRange: this._getCurrentDateRange(),
          activeFilters: this._getActiveFilters()
        },
        sorting: this._getCurrentSortState(),
        timestamp: new Date().toISOString()
      };
    },
        
    _getCurrentColumnConfig: function () {
      const oTable = this.byId('projectsTable');
      if (!oTable) {
        return this._getDefaultColumnConfig();
      }
            
      const aColumns = oTable.getColumns();
      return aColumns.map((oColumn, iIndex) => {
        return {
          index: iIndex,
          label: oColumn.getHeader().getText(),
          visible: oColumn.getVisible(),
          key: oColumn.getId().split('--').pop()
        };
      });
    },
        
    _getCurrentStatusFilter: function () {
      // Get current status filter value from filter model
      return ''; // Placeholder - would get from actual filter implementation
    },
        
    _getCurrentDateRange: function () {
      // Get current date range filter
      return null; // Placeholder - would get from actual filter implementation
    },
        
    _getActiveFilters: function () {
      // Get list of active filters
      return []; // Placeholder - would get from actual filter implementation
    },
        
    _getCurrentSortState: function () {
      const oTable = this.byId('projectsTable');
      if (!oTable || !oTable.getBinding('items')) {
        return { field: 'last_modified', descending: true };
      }
            
      const aBinding = oTable.getBinding('items');
      const aSorters = aBinding.aSorters || [];
            
      if (aSorters.length > 0) {
        return {
          field: aSorters[0].sPath,
          descending: aSorters[0].bDescending
        };
      }
            
      return { field: 'last_modified', descending: true };
    },
        
    onSaveCurrentView: function () {
      if (!this._oSaveViewDialog) {
        this._oSaveViewDialog = sap.ui.xmlfragment(
          'a2a.portal.fragment.SaveViewDialog', 
          this
        );
        this.getView().addDependent(this._oSaveViewDialog);
      }
            
      // Initialize dialog with current state
      const oCurrentState = this.getCurrentViewState();
      const oDialogModel = new JSONModel({
        name: '',
        description: '',
        setAsDefault: false,
        overwrite: false,
        existingViews: this._getSavedViewNames(),
        currentState: oCurrentState
      });
            
      this._oSaveViewDialog.setModel(oDialogModel, 'saveView');
      this._oSaveViewDialog.open();
    },
        
    onConfirmSaveView: function () {
      const oDialogModel = this._oSaveViewDialog.getModel('saveView');
      const sName = oDialogModel.getProperty('/name');
      const sDescription = oDialogModel.getProperty('/description');
      const bSetAsDefault = oDialogModel.getProperty('/setAsDefault');
      const oCurrentState = oDialogModel.getProperty('/currentState');
            
      if (!sName.trim()) {
        MessageToast.show('Please enter a view name');
        return;
      }
            
      // Create view object
      const oView = {
        id: this._generateViewId(),
        name: sName.trim(),
        description: sDescription.trim(),
        isDefault: bSetAsDefault,
        created: new Date().toISOString(),
        modified: new Date().toISOString(),
        ...oCurrentState
      };
            
      // Save view
      this._saveView(oView);
            
      MessageToast.show(`View '${  sName  }' saved successfully`);
      this._oSaveViewDialog.close();
    },
        
    onCancelSaveView: function () {
      this._oSaveViewDialog.close();
    },
        
    _generateViewId: function () {
      return `view_${  Date.now()  }_${  Math.random().toString(36).substr(2, 9)}`;
    },
        
    _saveView: function (oView) {
      const oSavedViewsModel = this.getView().getModel('savedViews');
      const aViews = oSavedViewsModel.getProperty('/views') || [];
            
      // Check if updating existing view
      const iExistingIndex = aViews.findIndex((v) => {
        return v.name === oView.name;
      });
            
      if (iExistingIndex >= 0) {
        // Update existing view
        oView.id = aViews[iExistingIndex].id;
        oView.created = aViews[iExistingIndex].created;
        aViews[iExistingIndex] = oView;
      } else {
        // Add new view
        aViews.push(oView);
      }
            
      // If setting as default, remove default from others
      if (oView.isDefault) {
        aViews.forEach((v) => {
          if (v.id !== oView.id) {
            v.isDefault = false;
          }
        });
      }
            
      oSavedViewsModel.setProperty('/views', aViews);
      this._persistSavedViews(aViews);
      this._updateSavedViewsMenu();
    },
        
    _loadSavedViews: function () {
      const sUserId = 'current_user';
      const sKey = `a2a_projects_saved_views_${  sUserId}`;
            
      try {
        const sStoredViews = localStorage.getItem(sKey);
        if (sStoredViews) {
          const aViews = JSON.parse(sStoredViews);
          const oSavedViewsModel = this.getView().getModel('savedViews');
          oSavedViewsModel.setProperty('/views', aViews);
          this._updateSavedViewsMenu();
        }
      } catch (e) {
        console.warn('Failed to load saved views:', e);
      }
    },
        
    _persistSavedViews: function (aViews) {
      const sUserId = 'current_user';
      const sKey = `a2a_projects_saved_views_${  sUserId}`;
            
      try {
        localStorage.setItem(sKey, JSON.stringify(aViews));
      } catch (e) {
        console.warn('Failed to save views:', e);
      }
            
      // Also save to backend
      this._saveSavedViewsToBackend(aViews);
    },
        
    _saveSavedViewsToBackend: function (aViews) {
      jQuery.ajax({
        url: '/api/user/preferences/saved-views',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
          view: 'projects',
          savedViews: aViews
        }),
        success: function () {
          // eslint-disable-next-line no-console
          // eslint-disable-next-line no-console
          console.log('Saved views synced to backend');
        },
        error: function () {
          console.warn('Failed to sync saved views to backend');
        }
      });
    },
        
    _updateSavedViewsMenu: function () {
      const oMenu = this.byId('savedViewsMenu');
      if (!oMenu) {
        return;
      }
            
      const oSavedViewsModel = this.getView().getModel('savedViews');
      const aViews = oSavedViewsModel.getProperty('/views') || [];
            
      // Remove existing view menu items (keep save/manage items)
      const aItems = oMenu.getItems();
      for (let i = aItems.length - 1; i >= 2; i--) {
        oMenu.removeItem(aItems[i]);
      }
            
      // Add separator if views exist
      if (aViews.length > 0) {
        oMenu.addItem(new sap.ui.unified.MenuItem({
          text: '---',
          enabled: false
        }));
      }
            
      // Add saved view menu items
      aViews.forEach((oView) => {
        oMenu.addItem(new sap.ui.unified.MenuItem({
          text: oView.name,
          icon: oView.isDefault ? 'sap-icon://favorite' : 'sap-icon://bookmark',
          press: this.onApplySavedView.bind(this),
          customData: [
            new sap.ui.core.CustomData({
              key: 'viewId',
              value: oView.id
            })
          ]
        }));
      });
    },
        
    onApplySavedView: function (oEvent) {
      const sViewId = oEvent.getSource().data('viewId');
      this._applyView(sViewId);
    },
        
    _applyView: function (sViewId) {
      const oSavedViewsModel = this.getView().getModel('savedViews');
      const aViews = oSavedViewsModel.getProperty('/views') || [];
            
      const oView = aViews.find((v) => {
        return v.id === sViewId; 
      });
      if (!oView) {
        MessageToast.show('View not found');
        return;
      }
            
      // Apply all view settings
      this._applyViewState(oView);
            
      // Update current view reference
      oSavedViewsModel.setProperty('/currentView', oView);
            
      MessageToast.show(`Applied view: ${  oView.name}`);
    },
        
    _applyViewState: function (oViewState) {
      // Apply view mode
      const oViewModel = this.getView().getModel('view');
      if (oViewModel) {
        oViewModel.setProperty('/viewMode', oViewState.viewMode || 'tiles');
      }
            
      // Apply density
      this._applyDensity(oViewState.density || 'cozy');
            
      // Apply column configuration
      if (oViewState.columnConfig) {
        this._applyColumnConfiguration(oViewState.columnConfig);
      }
            
      // Apply filters
      this._applyFilters(oViewState.filters || {});
            
      // Apply sorting
      this._applySorting(oViewState.sorting || {});
    },
        
    _applyFilters: function (oFilters) {
      // Apply search filter
      const oSearchField = this.byId('searchField');
      if (oSearchField && oFilters.search) {
        oSearchField.setValue(oFilters.search);
      }
            
      // Apply other filters - placeholder for actual filter implementation
      // Would apply status, date range, and other active filters
    },
        
    _applySorting: function (oSorting) {
      const oTable = this.byId('projectsTable');
      if (!oTable || !oTable.getBinding('items')) {
        return;
      }
            
      const oBinding = oTable.getBinding('items');
      if (oSorting.field) {
        const oSorter = new sap.ui.model.Sorter(oSorting.field, oSorting.descending);
        oBinding.sort([oSorter]);
      }
    },
        
    _applyLastUsedView: function () {
      const oSavedViewsModel = this.getView().getModel('savedViews');
      const aViews = oSavedViewsModel.getProperty('/views') || [];
            
      // Apply default view if one is set
      const oDefaultView = aViews.find((v) => {
        return v.isDefault; 
      });
      if (oDefaultView) {
        this._applyView(oDefaultView.id);
      }
    },
        
    _getSavedViewNames: function () {
      const oSavedViewsModel = this.getView().getModel('savedViews');
      const aViews = oSavedViewsModel.getProperty('/views') || [];
      return aViews.map((v) => {
        return v.name; 
      });
    },
        
    onManageSavedViews: function () {
      if (!this._oManageViewsDialog) {
        this._oManageViewsDialog = sap.ui.xmlfragment(
          'a2a.portal.fragment.ManageViewsDialog', 
          this
        );
        this.getView().addDependent(this._oManageViewsDialog);
      }
            
      // Initialize dialog with current views
      const oSavedViewsModel = this.getView().getModel('savedViews');
      const aViews = oSavedViewsModel.getProperty('/views') || [];
            
      const oManageModel = new JSONModel({
        views: JSON.parse(JSON.stringify(aViews)) // Deep copy
      });
            
      this._oManageViewsDialog.setModel(oManageModel, 'manageViews');
      this._oManageViewsDialog.open();
    },
        
    onDeleteSavedView: function (oEvent) {
      const oBindingContext = oEvent.getSource().getBindingContext('manageViews');
      const oView = oBindingContext.getObject();
            
      sap.m.MessageBox.confirm(
        `Are you sure you want to delete the view '${  oView.name  }'?`,
        {
          onClose: function (oAction) {
            if (oAction === sap.m.MessageBox.Action.OK) {
              this._deleteView(oView.id);
            }
          }.bind(this)
        }
      );
    },
        
    _deleteView: function (sViewId) {
      const oSavedViewsModel = this.getView().getModel('savedViews');
      const aViews = oSavedViewsModel.getProperty('/views') || [];
            
      const iIndex = aViews.findIndex((v) => {
        return v.id === sViewId; 
      });
      if (iIndex >= 0) {
        aViews.splice(iIndex, 1);
        oSavedViewsModel.setProperty('/views', aViews);
        this._persistSavedViews(aViews);
        this._updateSavedViewsMenu();
                
        MessageToast.show('View deleted successfully');
      }
    },
        
    onCloseManageViews: function () {
      this._oManageViewsDialog.close();
    },
        
    onTableSelectionChange: function (oEvent) {
      const oTable = oEvent.getSource();
      const aSelectedItems = oTable.getSelectedItems();
      const aSelectedContexts = oTable.getSelectedContexts();
            
      // Update view model with selection info
      const oViewModel = this.getView().getModel('view');
      oViewModel.setProperty('/selectedItemsCount', aSelectedItems.length);
      oViewModel.setProperty('/hasSelectedItems', aSelectedItems.length > 0);
            
      // Update toolbar button states
      this._updateBulkActionButtons(aSelectedItems.length);
            
      // Store selected projects data
      const aSelectedProjects = aSelectedContexts.map((oContext) => {
        return oContext.getObject();
      });
      oViewModel.setProperty('/selectedProjects', aSelectedProjects);
            
      // Update selection-dependent tooltips
      this._updateSelectionTooltips(aSelectedItems.length);
    },
        
    _updateBulkActionButtons: function (iSelectedCount) {
      const oExportButton = this.byId('exportSelectedButton');
      const oDeleteButton = this.byId('deleteSelectedButton');
      const oArchiveButton = this.byId('archiveSelectedButton');
      const oStatusButton = this.byId('bulkStatusButton');
            
      const bHasSelection = iSelectedCount > 0;
            
      // Update button enabled state
      if (oExportButton) {
        oExportButton.setEnabled(bHasSelection);
      }
      if (oDeleteButton) {
        oDeleteButton.setEnabled(bHasSelection);
      }
      if (oArchiveButton) {
        oArchiveButton.setEnabled(bHasSelection);
      }
      if (oStatusButton) {
        oStatusButton.setEnabled(bHasSelection);
      }
    },
        
    _updateSelectionTooltips: function (iSelectedCount) {
      const oExportButton = this.byId('exportSelectedButton');
      const oDeleteButton = this.byId('deleteSelectedButton');
            
      if (oExportButton) {
        const sExportTooltip = iSelectedCount > 0 ? 
          `Export ${  iSelectedCount  } selected projects` : 
          'Select projects to export';
        oExportButton.setTooltip(sExportTooltip);
      }
            
      if (oDeleteButton) {
        const sDeleteTooltip = iSelectedCount > 0 ? 
          `Delete ${  iSelectedCount  } selected projects` : 
          'Select projects to delete';
        oDeleteButton.setTooltip(sDeleteTooltip);
      }
    },
        
    onSelectAll: function () {
      const oTable = this.byId('projectsTable');
      oTable.selectAll();
    },
        
    onDeselectAll: function () {
      const oTable = this.byId('projectsTable');
      oTable.removeSelections();
    },
        
    onExportSelected: function () {
      const oViewModel = this.getView().getModel('view');
      const aSelectedProjects = oViewModel.getProperty('/selectedProjects') || [];
            
      if (aSelectedProjects.length === 0) {
        MessageToast.show('Please select projects to export');
        return;
      }
            
      // Show export options dialog
      this._showExportDialog(aSelectedProjects, false);
    },
        
        
    onConfirmExport: function () {
      const oExportModel = this._oExportDialog.getModel('export');
      const aSelectedProjects = oExportModel.getProperty('/selectedProjects');
      const sFormat = oExportModel.getProperty('/exportFormat');
      const bIncludeDetails = oExportModel.getProperty('/includeDetails');
      const bIncludeAgents = oExportModel.getProperty('/includeAgents');
      const bIncludeFiles = oExportModel.getProperty('/includeFiles');
      const sFileName = oExportModel.getProperty('/fileName');
            
      // Prepare export data
      const oExportData = {
        projects: aSelectedProjects,
        options: {
          format: sFormat,
          includeDetails: bIncludeDetails,
          includeAgents: bIncludeAgents,
          includeFiles: bIncludeFiles,
          fileName: sFileName
        }
      };
            
      this._performExport(oExportData);
      this._oExportDialog.close();
    },
        
    onCancelExport: function () {
      this._oExportDialog.close();
    },
        
    _performExport: function (oExportData) {
      const that = this;
            
      // Show progress indicator
      sap.ui.core.BusyIndicator.show();
            
      jQuery.ajax({
        url: '/api/projects/export',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify(oExportData),
        success: function (data, _textStatus, _xhr) {
          sap.ui.core.BusyIndicator.hide();
                    
          if (data.downloadUrl) {
            // Trigger download
            that._triggerDownload(data.downloadUrl, oExportData.options.fileName);
            MessageToast.show(`Export completed: ${  oExportData.projects.length  } projects`);
          } else {
            MessageToast.show('Export failed: No download URL received');
          }
        },
        error: function (xhr, status, error) {
          sap.ui.core.BusyIndicator.hide();
          MessageToast.show(`Export failed: ${  error}`);
        }
      });
    },
        
    _triggerDownload: function (sUrl, sFileName) {
      const oLink = document.createElement('a');
      oLink.href = sUrl;
      oLink.download = sFileName;
      document.body.appendChild(oLink);
      oLink.click();
      document.body.removeChild(oLink);
    },
        
    onDeleteSelected: function () {
      const oViewModel = this.getView().getModel('view');
      const aSelectedProjects = oViewModel.getProperty('/selectedProjects') || [];
            
      if (aSelectedProjects.length === 0) {
        MessageToast.show('Please select projects to delete');
        return;
      }
            
      // Show confirmation dialog
      const sMessage = `Are you sure you want to delete ${  aSelectedProjects.length  } selected projects?\n\nThis action cannot be undone.`;
            
      sap.m.MessageBox.confirm(sMessage, {
        title: 'Delete Projects',
        actions: [sap.m.MessageBox.Action.DELETE, sap.m.MessageBox.Action.CANCEL],
        emphasizedAction: sap.m.MessageBox.Action.DELETE,
        onClose: function (oAction) {
          if (oAction === sap.m.MessageBox.Action.DELETE) {
            this._performBulkDelete(aSelectedProjects);
          }
        }.bind(this)
      });
    },
        
    _performBulkDelete: function (aSelectedProjects) {
      const that = this;
      const aProjectIds = aSelectedProjects.map((project) => {
        return project.project_id;
      });
            
      // Show progress dialog
      this._showBulkProgressDialog('Deleting Projects', aProjectIds.length);
            
      // Perform bulk delete API call
      jQuery.ajax({
        url: '/api/projects/bulk-delete',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
          projectIds: aProjectIds
        }),
        success: function (data) {
          that._hideBulkProgressDialog();
                    
          if (data.deletedCount === aProjectIds.length) {
            MessageToast.show(`Successfully deleted ${  data.deletedCount  } projects`);
          } else {
            MessageToast.show(`Deleted ${  data.deletedCount  } of ${  aProjectIds.length  } projects`);
          }
                    
          // Refresh project list and clear selection
          that._loadProjects();
          that.onDeselectAll();
        },
        error: function (xhr, status, error) {
          that._hideBulkProgressDialog();
          MessageToast.show(`Bulk delete failed: ${  error}`);
        }
      });
    },
        
    onArchiveSelected: function () {
      const oViewModel = this.getView().getModel('view');
      const aSelectedProjects = oViewModel.getProperty('/selectedProjects') || [];
            
      if (aSelectedProjects.length === 0) {
        MessageToast.show('Please select projects to archive');
        return;
      }
            
      const sMessage = `Archive ${  aSelectedProjects.length  } selected projects?\n\nArchived projects can be restored later.`;
            
      sap.m.MessageBox.confirm(sMessage, {
        title: 'Archive Projects',
        onClose: function (oAction) {
          if (oAction === sap.m.MessageBox.Action.OK) {
            this._performBulkArchive(aSelectedProjects);
          }
        }.bind(this)
      });
    },
        
    _performBulkArchive: function (aSelectedProjects) {
      const that = this;
      const aProjectIds = aSelectedProjects.map((project) => {
        return project.project_id;
      });
            
      this._showBulkProgressDialog('Archiving Projects', aProjectIds.length);
            
      jQuery.ajax({
        url: '/api/projects/bulk-archive',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
          projectIds: aProjectIds
        }),
        success: function (data) {
          that._hideBulkProgressDialog();
          MessageToast.show(`Successfully archived ${  data.archivedCount  } projects`);
                    
          // Refresh project list and clear selection
          that._loadProjects();
          that.onDeselectAll();
        },
        error: function (xhr, status, error) {
          that._hideBulkProgressDialog();
          MessageToast.show(`Bulk archive failed: ${  error}`);
        }
      });
    },
        
    _showBulkProgressDialog: function (sTitle, iTotal) {
      if (!this._oBulkProgressDialog) {
        this._oBulkProgressDialog = new sap.m.BusyDialog({
          title: sTitle,
          text: `Processing ${  iTotal  } projects...`,
          showCancelButton: false
        });
      } else {
        this._oBulkProgressDialog.setTitle(sTitle);
        this._oBulkProgressDialog.setText(`Processing ${  iTotal  } projects...`);
      }
            
      this._oBulkProgressDialog.open();
    },
        
    _hideBulkProgressDialog: function () {
      if (this._oBulkProgressDialog) {
        this._oBulkProgressDialog.close();
      }
    },
        
    onBulkStatusChange: function () {
      const oViewModel = this.getView().getModel('view');
      const aSelectedProjects = oViewModel.getProperty('/selectedProjects') || [];
            
      if (aSelectedProjects.length === 0) {
        MessageToast.show('Please select projects to change status');
        return;
      }
            
      // Show status change dialog
      this._showBulkStatusDialog(aSelectedProjects);
    },
        
    _showBulkStatusDialog: function (aSelectedProjects) {
      if (!this._oBulkStatusDialog) {
        this._oBulkStatusDialog = sap.ui.xmlfragment(
          'a2a.portal.fragment.BulkStatusDialog', 
          this
        );
        this.getView().addDependent(this._oBulkStatusDialog);
      }
            
      const oStatusModel = new JSONModel({
        selectedProjects: aSelectedProjects,
        newStatus: 'active',
        statusOptions: [
          { key: 'active', text: 'Active' },
          { key: 'inactive', text: 'Inactive' },
          { key: 'maintenance', text: 'Maintenance' },
          { key: 'archived', text: 'Archived' }
        ]
      });
            
      this._oBulkStatusDialog.setModel(oStatusModel, 'bulkStatus');
      this._oBulkStatusDialog.open();
    },
        
    onConfirmBulkStatusChange: function () {
      const oStatusModel = this._oBulkStatusDialog.getModel('bulkStatus');
      const aSelectedProjects = oStatusModel.getProperty('/selectedProjects');
      const sNewStatus = oStatusModel.getProperty('/newStatus');
            
      this._performBulkStatusChange(aSelectedProjects, sNewStatus);
      this._oBulkStatusDialog.close();
    },
        
    onCancelBulkStatusChange: function () {
      this._oBulkStatusDialog.close();
    },
        
    _performBulkStatusChange: function (aSelectedProjects, sNewStatus) {
      const that = this;
      const aProjectIds = aSelectedProjects.map((project) => {
        return project.project_id;
      });
            
      this._showBulkProgressDialog('Updating Status', aProjectIds.length);
            
      jQuery.ajax({
        url: '/api/projects/bulk-status',
        method: 'POST',
        contentType: 'application/json',
        data: JSON.stringify({
          projectIds: aProjectIds,
          status: sNewStatus
        }),
        success: function (data) {
          that._hideBulkProgressDialog();
          MessageToast.show(`Status updated for ${  data.updatedCount  } projects`);
                    
          // Refresh project list and clear selection
          that._loadProjects();
          that.onDeselectAll();
        },
        error: function (xhr, status, error) {
          that._hideBulkProgressDialog();
          MessageToast.show(`Bulk status change failed: ${  error}`);
        }
      });
    },
        
    onExportAll: function () {
      const oViewModel = this.getView().getModel('view');
      const aAllProjects = oViewModel.getProperty('/projects') || [];
            
      if (aAllProjects.length === 0) {
        MessageToast.show('No projects available to export');
        return;
      }
            
      // Show export dialog with all projects
      this._showExportDialog(aAllProjects, true);
    },
        
    _showExportDialog: function (aProjects, bIsExportAll) {
      if (!this._oExportDialog) {
        this._oExportDialog = sap.ui.xmlfragment(
          'a2a.portal.fragment.ExportDialog', 
          this
        );
        this.getView().addDependent(this._oExportDialog);
      }
            
      const sDefaultFileName = bIsExportAll ? 
        `all_projects_${  new Date().toISOString().split('T')[0]}` :
        `selected_projects_${  new Date().toISOString().split('T')[0]}`;
            
      // Initialize export dialog model
      const oExportModel = new JSONModel({
        selectedProjects: aProjects,
        exportFormat: 'excel',
        includeDetails: true,
        includeAgents: false,
        includeFiles: false,
        fileName: sDefaultFileName,
        isExportAll: bIsExportAll || false,
        exportScope: bIsExportAll ? 'all' : 'selected'
      });
            
      this._oExportDialog.setModel(oExportModel, 'export');
      this._oExportDialog.open();
    },
        
    onExportTemplateDownload: function () {
      // Download empty template for project import
      this._triggerDownload('/templates/project_import_template.xlsx', 'project_import_template.xlsx');
      MessageToast.show('Download started: Project import template');
    },
        
    onExportWithFilters: function () {
      // Export only visible/filtered projects
      const oTable = this.byId('projectsTable');
      const oBinding = oTable.getBinding('items');
      let aFilteredProjects = [];
            
      if (oBinding) {
        const aContexts = oBinding.getContexts();
        aFilteredProjects = aContexts.map((oContext) => {
          return oContext.getObject();
        });
      }
            
      if (aFilteredProjects.length === 0) {
        MessageToast.show('No projects match current filters');
        return;
      }
            
      // Show export dialog with filtered projects
      const sFileName = `filtered_projects_${  new Date().toISOString().split('T')[0]}`;
      this._showExportDialogWithCustomName(aFilteredProjects, sFileName, 'filtered');
    },
        
    _showExportDialogWithCustomName: function (aProjects, sFileName, sScope) {
      if (!this._oExportDialog) {
        this._oExportDialog = sap.ui.xmlfragment(
          'a2a.portal.fragment.ExportDialog', 
          this
        );
        this.getView().addDependent(this._oExportDialog);
      }
            
      const oExportModel = new JSONModel({
        selectedProjects: aProjects,
        exportFormat: 'excel',
        includeDetails: true,
        includeAgents: false,
        includeFiles: false,
        fileName: sFileName,
        isExportAll: false,
        exportScope: sScope
      });
            
      this._oExportDialog.setModel(oExportModel, 'export');
      this._oExportDialog.open();
    },
        
    onFormatChange: function (oEvent) {
      const oExportModel = this._oExportDialog.getModel('export');
      const sSelectedFormat = oEvent.getSource().getText().toLowerCase();
      let sFormat = 'excel';
      let sExtension = '.xlsx';
            
      if (sSelectedFormat.includes('csv')) {
        sFormat = 'csv';
        sExtension = '.csv';
      } else if (sSelectedFormat.includes('json')) {
        sFormat = 'json';
        sExtension = '.json';
      }
            
      oExportModel.setProperty('/exportFormat', sFormat);
            
      // Update filename extension
      const sCurrentFileName = oExportModel.getProperty('/fileName');
      const sBaseFileName = sCurrentFileName.replace(/\.[^/.]+$/, '');
      oExportModel.setProperty('/fileName', sBaseFileName + sExtension);
    },

    onToggleFavorite: function (oEvent) {
      const oBindingContext = oEvent.getSource().getBindingContext('view');
      const oProject = oBindingContext.getObject();
      const bIsFavorite = oProject.isFavorite || false;
            
      // Toggle favorite status
      oProject.isFavorite = !bIsFavorite;
      oBindingContext.getModel().updateBindings();
            
      const sMessage = bIsFavorite ? 'Removed from favorites' : 'Added to favorites';
      MessageToast.show(`${sMessage  }: ${  oProject.name}`);
    },
        
    onShareProject: function (oEvent) {
      const oBindingContext = oEvent.getSource().getBindingContext('view');
      const oProject = oBindingContext.getObject();
            
      // Copy share link to clipboard
      const sShareUrl = `${window.location.origin  }/projects/${  oProject.project_id}`;
            
      if (navigator.clipboard) {
        navigator.clipboard.writeText(sShareUrl).then(() => {
          MessageToast.show('Share link copied to clipboard');
        });
      } else {
        MessageToast.show(`Share link: ${  sShareUrl}`);
      }
    },
        
    onExit: function () {
      // Clean up hover event handlers
      const oTable = this.byId('projectsTable');
      if (oTable && oTable.getDomRef()) {
        jQuery(oTable.getDomRef()).off('mouseenter mouseleave', '.a2a-project-row');
      }
    },

    onOpenSortDialog: function () {
      if (!this._oSortDialog) {
        this._oSortDialog = sap.ui.xmlfragment('a2a.portal.fragment.SortDialog', this);
        this.getView().addDependent(this._oSortDialog);
      }
      this._oSortDialog.open();
    },

    onSortConfirm: function (oEvent) {
      const oSortItem = oEvent.getParameter('sortItem');
      const bDescending = oEvent.getParameter('sortDescending');
      const oTable = this.byId('projectsTable');
      const oBinding = oTable.getBinding('items');
            
      if (oSortItem) {
        const sSortPath = oSortItem.getKey();
        const oSorter = new sap.ui.model.Sorter(sSortPath, bDescending);
        oBinding.sort(oSorter);
      }
    },

    onEditProject: function (oEvent) {
      const oContext = oEvent.getSource().getBindingContext('view');
      const oProjectData = oContext.getProperty();
            
      if (!this._oEditDialog) {
        this._oEditDialog = sap.ui.xmlfragment('a2a.portal.fragment.EditProjectDialog', this);
        this.getView().addDependent(this._oEditDialog);
      }
            
      // Create a JSON model for the dialog
      const oDialogModel = new JSONModel(JSON.parse(JSON.stringify(oProjectData)));
      this._oEditDialog.setModel(oDialogModel);
      this._oEditDialog.open();
    },

    onEditProjectConfirm: function (oEvent) {
      const oDialog = oEvent.getSource().getParent();
      const oData = oDialog.getModel().getData();
            
      // Here you would normally save the changes to the backend
      MessageToast.show(`Project updated: ${  oData.name}`);
      this._loadProjects();
      oDialog.close();
    },

    onEditProjectCancel: function (oEvent) {
      oEvent.getSource().getParent().close();
    },

    onCloneProject: function (oEvent) {
      const oContext = oEvent.getSource().getBindingContext('view');
      const sProjectName = oContext.getProperty('name');
            
      MessageBox.confirm(
        `Clone project '${  sProjectName  }'?`, {
          onClose: function (sAction) {
            if (sAction === MessageBox.Action.OK) {
              // Implement clone logic
              MessageToast.show('Clone functionality - coming soon');
            }
          }
        }
      );
    },

    onArchiveProject: function (oEvent) {
      const oContext = oEvent.getSource().getBindingContext('view');
      const sProjectName = oContext.getProperty('name');
      const sProjectId = oContext.getProperty('project_id');
            
      MessageBox.confirm(
        `Archive project '${  sProjectName  }'? This will move the project to archived status and hide it from the active projects list.`, {
          icon: MessageBox.Icon.QUESTION,
          title: 'Archive Project',
          onClose: function (sAction) {
            if (sAction === MessageBox.Action.OK) {
              this._archiveProject(sProjectId, sProjectName);
            }
          }.bind(this)
        }
      );
    },

    _archiveProject: function (sProjectId, sProjectName) {
      jQuery.ajax({
        url: `/api/projects/${  sProjectId  }/archive`,
        method: 'POST',
        success: function () {
          MessageToast.show(`Project '${  sProjectName  }' archived successfully`);
          this._loadProjects();
        }.bind(this),
        error: function (xhr, _status, _error) {
          let sMessage = 'Failed to archive project';
          if (xhr.responseJSON && xhr.responseJSON.detail) {
            sMessage += `: ${  xhr.responseJSON.detail}`;
          }
          MessageToast.show(sMessage);
        }.bind(this)
      });
    },

    onDeleteProject: function (oEvent) {
      const oContext = oEvent.getSource().getBindingContext('view');
      const sProjectName = oContext.getProperty('name');
      const sProjectId = oContext.getProperty('project_id');
            
      MessageBox.confirm(
        `Delete project '${  sProjectName  }'? This action cannot be undone.`, {
          icon: MessageBox.Icon.WARNING,
          title: 'Confirm Deletion',
          onClose: function (sAction) {
            if (sAction === MessageBox.Action.OK) {
              this._deleteProject(sProjectId);
            }
          }.bind(this)
        }
      );
    },

    _deleteProject: function (sProjectId) {
      jQuery.ajax({
        url: `/api/projects/${  sProjectId}`,
        method: 'DELETE',
        success: function () {
          MessageToast.show('Project deleted successfully');
          this._loadProjects();
        }.bind(this),
        error: function (_xhr, _status, error) {
          MessageToast.show(`Failed to delete project: ${  error}`);
        }.bind(this)
      });
    },

    // Duplicate method - commented out
    // onExportSelected: function () {
    //     const oTable = this.byId("projectsTable");
    //     const aSelectedItems = oTable.getSelectedItems();
    //     
    //     if (aSelectedItems.length === 0) {
    //         MessageToast.show("Please select projects to export");
    //         return;
    //     }
    //     
    //     MessageToast.show("Export functionality - coming soon");
    // },

    // Duplicate method - commented out  
    // onDeleteSelected: function () {
    //     const oTable = this.byId("projectsTable");
    //     const aSelectedItems = oTable.getSelectedItems();
    //     
    //     if (aSelectedItems.length === 0) {
    //         MessageToast.show("Please select projects to delete");
    //         return;
    //     }
    //     
    //     const sMessage = `Delete ${  aSelectedItems.length  } selected project(s)? This action cannot be undone.`;
    //     
    //     MessageBox.confirm(sMessage, {
    //         icon: MessageBox.Icon.WARNING,
    //         title: "Confirm Deletion",
    //         onClose: function (sAction) {
    //             if (sAction === MessageBox.Action.OK) {
    //                 // Implement batch delete
    //                 MessageToast.show("Batch delete functionality - coming soon");
    //             }
    //         }
    //     });
    // },

    onCreateFromTemplate: function () {
      MessageToast.show('Create from template - coming soon');
    },

    onViewTutorials: function () {
      MessageToast.show('Opening tutorials...');
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