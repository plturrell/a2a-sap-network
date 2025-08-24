sap.ui.define([
  'sap/a2a/controller/BaseController',
  'sap/ui/model/json/JSONModel',
  'sap/ui/model/Filter',
  'sap/ui/model/FilterOperator',
  'sap/m/MessageToast',
  'sap/m/MessageBox',
  'sap/ui/core/Fragment',
  'sap/ui/export/Spreadsheet',
  'sap/ui/export/library',
  'sap/m/Table',
  'sap/m/Column',
  'sap/m/ColumnListItem',
  'sap/m/Text',
  'sap/m/Title',
  'sap/m/VBox',
  'sap/m/HBox'
], (BaseController, JSONModel, Filter, FilterOperator, MessageToast, MessageBox, Fragment, Spreadsheet, exportLibrary, Table, Column, ColumnListItem, Text, Title, VBox, HBox) => {
  'use strict';
  /* global URL, Blob, Worker */

  /**
     * SAP Fiori Elements List Report Controller for Projects
     * Implements enterprise patterns with smart controls and responsive design
     */
  return BaseController.extend('sap.a2a.controller.ProjectsListReport', {

    /* =========================================================== */
    /* lifecycle methods                                           */
    /* =========================================================== */

    /**
         * Called when the controller is instantiated.
         */
    onInit: function () {
      // eslint-disable-next-line prefer-rest-params
      BaseController.prototype.onInit.apply(this, arguments);
            
      this._oViewModel = new JSONModel({
        busy: false,
        delay: 0,
        selectedProjects: [],
        totalProjects: 0,
        kpi: {
          totalProjects: 0,
          activeProjects: 0,
          completedProjects: 0,
          deploymentSuccessRate: 0
        },
        viewMode: 'table',
        filterExpanded: false
      });
            
      this.setModel(this._oViewModel, 'listReport');
            
      // Initialize filter and sort state
      this._mFilters = {
        search: [],
        status: [],
        priority: [],
        businessUnit: []
      };
            
      // Setup router event handling
      this.getRouter().getRoute('projects').attachPatternMatched(this._onPatternMatched, this);
            
      // Load initial data
      this._loadKPIData();
      this._loadProjectStatuses();
      this._loadProjectPriorities();
            
      // Setup help system integration
      this._setupContextualHelp();
            
      // Initialize column sorting configuration
      this._initializeColumnSorting();
            
      // Initialize aggregation configuration
      this._initializeAggregations();
    },

    /* =========================================================== */
    /* event handlers                                              */
    /* =========================================================== */

    /**
         * Triggered when route pattern is matched
         */
    _onPatternMatched: function () {
      this._refreshData();
    },

    /**
         * Handles create project button press
         */
    onCreateProject: function () {
      this.getRouter().navTo('projectCreate');
    },

    /**
         * Handles project item press - navigation to object page
         */
    onProjectPress: function (oEvent) {
      const oBindingContext = oEvent.getSource().getBindingContext();
      const sProjectId = oBindingContext.getProperty('id');
            
      this.getRouter().navTo('projectDetail', {
        projectId: sProjectId
      });
    },

    /**
         * Handles refresh button press
         */
    onRefresh: function () {
      this._refreshData();
      this._loadKPIData();
      MessageToast.show(this.getResourceBundle().getText('common.refreshed'));
    },

    /**
         * Handles filter bar search
         */
    onFilterBarSearch: function () {
      this._applyFilters();
    },

    /**
         * Handles filter change events
         */
    onFilterChange: function () {
      this._applyFilters();
    },

    /**
         * Handles row selection change
         */
    onRowSelectionChange: function (oEvent) {
      const aSelectedItems = oEvent.getSource().getSelectedItems();
      const aSelectedProjects = aSelectedItems.map((oItem) => {
        return oItem.getBindingContext().getObject();
      });
            
      this._oViewModel.setProperty('/selectedProjects', aSelectedProjects);
            
      // Update bulk action button states
      this._updateBulkActionButtons(aSelectedProjects.length > 0);
    },

    /**
         * Handles before rebind table event for smart table
         */
    onBeforeRebindTable: function (oEvent) {
      const mBindingParams = oEvent.getParameter('bindingParams');
            
      // Add custom filters
      Object.keys(this._mFilters).forEach(sFilterKey => {
        if (this._mFilters[sFilterKey].length > 0) {
          mBindingParams.filters = mBindingParams.filters.concat(this._mFilters[sFilterKey]);
        }
      });
            
      // Add default sorting
      if (!mBindingParams.sorter || mBindingParams.sorter.length === 0) {
        mBindingParams.sorter = [
          new sap.ui.model.Sorter('modifiedAt', true) // Sort by modified date, descending
        ];
      }
    },

    /**
         * Handles KPI tile press
         */
    onKPIPress: function (oEvent) {
      const sTileId = oEvent.getSource().getId();
      let sFilter = '';
            
      if (sTileId.includes('active')) {
        sFilter = "status eq 'ACTIVE'";
      } else if (sTileId.includes('completed')) {
        sFilter = "status eq 'COMPLETED'";
      }
            
      if (sFilter) {
        this._applyQuickFilter(sFilter);
      }
    },

    /**
         * Handles view mode change (table/cards)
         */
    onViewModeChange: function (oEvent) {
      const sSelectedKey = oEvent.getParameter('key');
      this._oViewModel.setProperty('/viewMode', sSelectedKey);
            
      if (sSelectedKey === 'cards') {
        this._switchToCardView();
      } else {
        this._switchToTableView();
      }
    },

    /**
         * Handles bulk edit operation
         */
    onBulkEdit: function () {
      const aSelectedProjects = this._oViewModel.getProperty('/selectedProjects');
            
      if (aSelectedProjects.length === 0) {
        MessageToast.show(this.getResourceBundle().getText('projects.noSelection'));
        return;
      }
            
      // Open bulk edit dialog
      this._openBulkEditDialog(aSelectedProjects);
    },

    /**
         * Handles bulk delete operation
         */
    onBulkDelete: function () {
      const aSelectedProjects = this._oViewModel.getProperty('/selectedProjects');
            
      if (aSelectedProjects.length === 0) {
        MessageToast.show(this.getResourceBundle().getText('projects.noSelection'));
        return;
      }
            
      const sConfirmText = this.getResourceBundle().getText('projects.deleteConfirm', [aSelectedProjects.length]);
            
      MessageBox.confirm(sConfirmText, {
        title: this.getResourceBundle().getText('projects.deleteTitle'),
        onClose: (sAction) => {
          if (sAction === MessageBox.Action.OK) {
            this._deleteBulkProjects(aSelectedProjects);
          }
        },
        emphasizedAction: MessageBox.Action.OK
      });
    },

    /**
         * Handles project deployment
         */
    onDeployProject: function (oEvent) {
      const oBindingContext = oEvent.getSource().getBindingContext();
      const oProject = oBindingContext.getObject();
            
      this._deployProject(oProject);
    },

    /**
         * Handles project stop
         */
    onStopProject: function (oEvent) {
      const oBindingContext = oEvent.getSource().getBindingContext();
      const oProject = oBindingContext.getObject();
            
      this._stopProject(oProject);
    },

    /**
         * Handles edit project
         */
    onEditProject: function (oEvent) {
      const oBindingContext = oEvent.getSource().getBindingContext();
      const sProjectId = oBindingContext.getProperty('id');
            
      this.getRouter().navTo('projectEdit', {
        projectId: sProjectId
      });
    },

    /**
         * Handles clone project
         */
    onCloneProject: function (oEvent) {
      const oBindingContext = oEvent.getSource().getBindingContext();
      const oProject = oBindingContext.getObject();
            
      this._cloneProject(oProject);
    },

    /**
         * Handles export projects
         */
    onExportProjects: function () {
      this._exportToExcel();
    },

    /**
         * Handles export single project
         */
    onExportProject: function (oEvent) {
      const oBindingContext = oEvent.getSource().getBindingContext();
      const oProject = oBindingContext.getObject();
            
      this._exportSingleProject(oProject);
    },

    /**
         * Handles delete project
         */
    onDeleteProject: function (oEvent) {
      const oBindingContext = oEvent.getSource().getBindingContext();
      const oProject = oBindingContext.getObject();
            
      const sConfirmText = this.getResourceBundle().getText('projects.deleteProjectConfirm', [oProject.name]);
            
      MessageBox.confirm(sConfirmText, {
        title: this.getResourceBundle().getText('projects.deleteTitle'),
        onClose: (sAction) => {
          if (sAction === MessageBox.Action.OK) {
            this._deleteProject(oProject);
          }
        },
        emphasizedAction: MessageBox.Action.OK
      });
    },

    /**
         * Handles import projects
         */
    onImportProjects: function () {
      if (!this._pImportDialog) {
        this._pImportDialog = Fragment.load({
          id: this.getView().getId(),
          name: 'sap.a2a.view.fragments.ImportProjectsDialog',
          controller: this
        }).then((oDialog) => {
          this.getView().addDependent(oDialog);
          return oDialog;
        });
      }
            
      this._pImportDialog.then((oDialog) => {
        oDialog.open();
      });
    },

    /**
         * Handles manage templates
         */
    onManageTemplates: function () {
      this.getRouter().navTo('templates');
    },

    /**
         * Handles value help for business unit filter
         */
    onBusinessUnitValueHelp: function (oEvent) {
      this._openBusinessUnitValueHelp(oEvent.getSource());
    },

    /**
         * Handles settings dialog
         */
    onOpenSettings: function () {
      if (!this._pSettingsDialog) {
        this._pSettingsDialog = Fragment.load({
          id: this.getView().getId(),
          name: 'sap.a2a.view.fragments.ListReportSettingsDialog',
          controller: this
        }).then((oDialog) => {
          this.getView().addDependent(oDialog);
          return oDialog;
        });
      }
            
      this._pSettingsDialog.then((oDialog) => {
        oDialog.open();
      });
    },

    /**
         * Handles contextual help
         */
    onOpenHelp: function () {
      const helpProvider = sap.a2a.utils.HelpProvider;
      helpProvider.showDetailedHelp(this.getView(), 'projects_list_report', {
        title: this.getResourceBundle().getText('help.projects.title'),
        placement: 'PreferredBottomOrFlip'
      });
    },

    /* =========================================================== */
    /* formatter functions                                         */
    /* =========================================================== */

    /**
         * Formats status to state for ObjectStatus
         */
    formatStatusState: function (sStatus) {
      const mStatusStates = {
        'DRAFT': 'None',
        'ACTIVE': 'Success',
        'TESTING': 'Warning',
        'DEPLOYED': 'Success',
        'PAUSED': 'Warning',
        'COMPLETED': 'Success',
        'ARCHIVED': 'None',
        'ERROR': 'Error'
      };
            
      return mStatusStates[sStatus] || 'None';
    },

    /**
         * Formats status to icon
         */
    formatStatusIcon: function (sStatus) {
      const mStatusIcons = {
        'DRAFT': 'sap-icon://document',
        'ACTIVE': 'sap-icon://play',
        'TESTING': 'sap-icon://lab',
        'DEPLOYED': 'sap-icon://cloud',
        'PAUSED': 'sap-icon://pause',
        'COMPLETED': 'sap-icon://complete',
        'ARCHIVED': 'sap-icon://archive',
        'ERROR': 'sap-icon://error'
      };
            
      return mStatusIcons[sStatus] || 'sap-icon://question-mark';
    },

    /**
         * Formats priority to state for ObjectStatus
         */
    formatPriorityState: function (sPriority) {
      const mPriorityStates = {
        'LOW': 'None',
        'MEDIUM': 'Warning',
        'HIGH': 'Error',
        'CRITICAL': 'Error'
      };
            
      return mPriorityStates[sPriority] || 'None';
    },

    /**
         * Formats priority to icon
         */
    formatPriorityIcon: function (sPriority) {
      const mPriorityIcons = {
        'LOW': 'sap-icon://down',
        'MEDIUM': 'sap-icon://right',
        'HIGH': 'sap-icon://up',
        'CRITICAL': 'sap-icon://alert'
      };
            
      return mPriorityIcons[sPriority];
    },

    /**
         * Formats priority to highlight for list items
         */
    formatPriorityHighlight: function (sPriority) {
      const mPriorityHighlights = {
        'CRITICAL': 'Error',
        'HIGH': 'Warning',
        'MEDIUM': 'Information',
        'LOW': 'None'
      };
            
      return mPriorityHighlights[sPriority] || 'None';
    },

    /**
         * Formats relative time for dates
         */
    formatRelativeTime: function (dDate) {
      if (!dDate) {
        return this.getResourceBundle().getText('common.never');
      }
            
      const oFormat = sap.ui.core.format.DateFormat.getDateTimeInstance({
        relative: true,
        relativeRange: [1, 30]
      });
            
      return oFormat.format(new Date(dDate));
    },

    /**
         * Formats deployment status
         */
    formatDeploymentStatus: function (sStatus) {
      if (!sStatus) {
        return '';
      }
            
      const mStatusTexts = {
        'SUCCESS': this.getResourceBundle().getText('projects.deploymentSuccess'),
        'FAILED': this.getResourceBundle().getText('projects.deploymentFailed'),
        'IN_PROGRESS': this.getResourceBundle().getText('projects.deploymentInProgress')
      };
            
      return mStatusTexts[sStatus] || sStatus;
    },

    /**
         * Formats percentage values
         */
    formatPercentage: function (nValue) {
      if (nValue === null || nValue === undefined) {
        return '0%';
      }
      return `${Math.round(nValue)  }%`;
    },

    /**
         * Determines if project is deployable
         */
    isDeployable: function (sStatus) {
      return sStatus === 'DRAFT' || sStatus === 'TESTING' || sStatus === 'PAUSED';
    },

    /**
         * Determines if project is stoppable
         */
    isStoppable: function (sStatus) {
      return sStatus === 'ACTIVE' || sStatus === 'DEPLOYED';
    },

    /* =========================================================== */
    /* internal methods                                            */
    /* =========================================================== */

    /**
         * Loads KPI data from backend
         */
    _loadKPIData: function () {
      this._oViewModel.setProperty('/busy', true);
            
      // Mock KPI data - replace with actual service call
      setTimeout(() => {
        this._oViewModel.setData({
          ...this._oViewModel.getData(),
          kpi: {
            totalProjects: 127,
            activeProjects: 45,
            completedProjects: 23,
            deploymentSuccessRate: 94.7
          },
          busy: false
        });
      }, 500);
    },

    /**
         * Loads project status options
         */
    _loadProjectStatuses: function () {
      const aStatuses = [
        { key: 'DRAFT', text: this.getResourceBundle().getText('projects.status.draft') },
        { key: 'ACTIVE', text: this.getResourceBundle().getText('projects.status.active') },
        { key: 'TESTING', text: this.getResourceBundle().getText('projects.status.testing') },
        { key: 'DEPLOYED', text: this.getResourceBundle().getText('projects.status.deployed') },
        { key: 'PAUSED', text: this.getResourceBundle().getText('projects.status.paused') },
        { key: 'COMPLETED', text: this.getResourceBundle().getText('projects.status.completed') },
        { key: 'ARCHIVED', text: this.getResourceBundle().getText('projects.status.archived') }
      ];
            
      this.getModel().setProperty('/ProjectStatuses', aStatuses);
    },

    /**
         * Loads project priority options
         */
    _loadProjectPriorities: function () {
      const aPriorities = [
        { key: 'LOW', text: this.getResourceBundle().getText('projects.priority.low') },
        { key: 'MEDIUM', text: this.getResourceBundle().getText('projects.priority.medium') },
        { key: 'HIGH', text: this.getResourceBundle().getText('projects.priority.high') },
        { key: 'CRITICAL', text: this.getResourceBundle().getText('projects.priority.critical') }
      ];
            
      this.getModel().setProperty('/ProjectPriorities', aPriorities);
    },

    /**
         * Refreshes main data
         */
    _refreshData: function () {
      const oSmartTable = this.byId('projectsSmartTable');
      if (oSmartTable) {
        oSmartTable.rebindTable();
      }
    },

    /**
         * Applies current filters to the table
         */
    _applyFilters: function () {
      const oSmartTable = this.byId('projectsSmartTable');
      if (oSmartTable) {
        oSmartTable.rebindTable();
      }
    },

    /**
         * Applies quick filter
         */
    _applyQuickFilter: function (sFilter) {
      const oSmartTable = this.byId('projectsSmartTable');
      const oTable = oSmartTable.getTable();
            
      // Create filter from OData filter string
      const aFilters = [new Filter({
        path: sFilter.split(' ')[0],
        operator: FilterOperator.EQ,
        value1: sFilter.split("'")[1]
      })];
            
      oTable.getBinding('items').filter(aFilters);
    },

    /**
         * Updates bulk action button states
         */
    _updateBulkActionButtons: function (bEnabled) {
      this.byId('bulkEditButton').setEnabled(bEnabled);
      this.byId('bulkDeleteButton').setEnabled(bEnabled);
            
      const sSelectionText = bEnabled ? 
        this.getResourceBundle().getText('projects.selectedCount', [this._oViewModel.getProperty('/selectedProjects').length]) : 
        '';
            
      this.byId('selectionInfo').setText(sSelectionText).setVisible(bEnabled);
    },

    /**
         * Switches to card view layout
         */
    _switchToCardView: function () {
      // Implementation for card view layout
      MessageToast.show(this.getResourceBundle().getText('common.cardViewNotImplemented'));
    },

    /**
         * Switches to table view layout
         */
    _switchToTableView: function () {
      // Already in table view
    },

    /**
         * Deploys a project
         */
    _deployProject: function (oProject) {
      this.showBusyIndicator();
            
      // Mock deployment - replace with actual service call
      setTimeout(() => {
        this.hideBusyIndicator();
        MessageToast.show(this.getResourceBundle().getText('projects.deploymentStarted', [oProject.name]));
        this._refreshData();
      }, 2000);
    },

    /**
         * Stops a project
         */
    _stopProject: function (oProject) {
      this.showBusyIndicator();
            
      // Mock stop - replace with actual service call
      setTimeout(() => {
        this.hideBusyIndicator();
        MessageToast.show(this.getResourceBundle().getText('projects.projectStopped', [oProject.name]));
        this._refreshData();
      }, 1000);
    },

    /**
         * Clones a project
         */
    _cloneProject: function (oProject) {
      this.getRouter().navTo('projectCreate', {}, {
        templateId: oProject.id
      });
    },

    /**
         * Deletes a single project
         */
    _deleteProject: function (oProject) {
      this.showBusyIndicator();
            
      // Mock delete - replace with actual service call
      setTimeout(() => {
        this.hideBusyIndicator();
        MessageToast.show(this.getResourceBundle().getText('projects.projectDeleted', [oProject.name]));
        this._refreshData();
        this._loadKPIData();
      }, 1000);
    },

    /**
         * Deletes multiple projects
         */
    _deleteBulkProjects: function (aProjects) {
      this.showBusyIndicator();
            
      // Mock bulk delete - replace with actual service call
      setTimeout(() => {
        this.hideBusyIndicator();
        MessageToast.show(this.getResourceBundle().getText('projects.projectsDeleted', [aProjects.length]));
        this._refreshData();
        this._loadKPIData();
                
        // Clear selection
        this._oViewModel.setProperty('/selectedProjects', []);
        this._updateBulkActionButtons(false);
      }, 2000);
    },

    /**
         * Opens bulk edit dialog
         */
    _openBulkEditDialog: function (aProjects) {
      if (!this._pBulkEditDialog) {
        this._pBulkEditDialog = Fragment.load({
          id: this.getView().getId(),
          name: 'sap.a2a.view.fragments.BulkEditProjectsDialog',
          controller: this
        }).then((oDialog) => {
          this.getView().addDependent(oDialog);
          return oDialog;
        });
      }
            
      this._pBulkEditDialog.then((oDialog) => {
        oDialog.getModel('bulkEdit').setProperty('/selectedProjects', aProjects);
        oDialog.open();
      });
    },

    /**
         * Exports data to Excel
         */
    _exportToExcel: function () {
      const aColumns = [
        { property: 'name', label: this.getResourceBundle().getText('projects.name') },
        { property: 'status', label: this.getResourceBundle().getText('projects.status') },
        { property: 'priority', label: this.getResourceBundle().getText('projects.priority') },
        { property: 'createdBy', label: this.getResourceBundle().getText('projects.createdBy') },
        { property: 'createdAt', label: this.getResourceBundle().getText('projects.createdAt'), type: exportLibrary.EdmType.DateTime },
        { property: 'modifiedAt', label: this.getResourceBundle().getText('projects.modifiedAt'), type: exportLibrary.EdmType.DateTime }
      ];
            
      const oSpreadsheet = new Spreadsheet({
        workbook: {
          columns: aColumns
        },
        dataSource: this.getModel().getProperty('/projects'),
        fileName: 'A2A_Projects_Export'
      });
            
      oSpreadsheet.build();
    },

    /**
         * Exports single project data
         */
    _exportSingleProject: function (oProject) {
      // Implementation for single project export
      MessageToast.show(this.getResourceBundle().getText('projects.exportStarted', [oProject.name]));
    },

    /**
         * Opens business unit value help
         */
    _openBusinessUnitValueHelp: function (_oSource) {
      // Implementation for business unit value help
      MessageToast.show(this.getResourceBundle().getText('common.valueHelpNotImplemented'));
    },

    /**
         * Handles table update finished event
         */
    onTableUpdateFinished: function (oEvent) {
      const iTotalItems = oEvent.getParameter('total') || oEvent.getSource().getItems().length;
      this._oViewModel.setProperty('/totalProjects', iTotalItems);
            
      // Update sort indicators after table refresh
      setTimeout(() => {
        if (this._oSortingModel) {
          this._updateSortIndicators();
        }
      }, 100);
    },
        
    /**
         * Sets up contextual help system
         */
    _setupContextualHelp: function () {
      const helpProvider = sap.a2a.utils.HelpProvider;
            
      // Enable help for key controls
      helpProvider.enableHelp(this.byId('createProjectButton'), 'projects.create');
      helpProvider.enableHelp(this.byId('listReportFilter'), 'projects.filter');
      helpProvider.enableHelp(this.byId('projectsSmartTable'), 'projects.table');
    },
        
    /**
         * Initializes advanced column sorting configuration
         */
    _initializeColumnSorting: function () {
      this._oSortingModel = new JSONModel({
        currentSort: {
          column: 'modifiedAt',
          direction: 'desc',
          active: true
        },
        availableSorts: [
          { key: 'name', text: 'Project Name', dataType: 'string', icon: 'sap-icon://alphabetical-order' },
          { key: 'status', text: 'Status', dataType: 'string', icon: 'sap-icon://status-positive' },
          { key: 'priority', text: 'Priority', dataType: 'string', icon: 'sap-icon://priority' },
          { key: 'createdAt', text: 'Created Date', dataType: 'date', icon: 'sap-icon://calendar' },
          { key: 'modifiedAt', text: 'Modified Date', dataType: 'date', icon: 'sap-icon://history' },
          { key: 'lastDeployment', text: 'Last Deployment', dataType: 'date', icon: 'sap-icon://cloud' },
          { key: 'createdBy', text: 'Created By', dataType: 'string', icon: 'sap-icon://person-placeholder' }
        ],
        sortHistory: [],
        quickSorts: [
          { key: 'newest', text: 'Newest First', column: 'createdAt', direction: 'desc' },
          { key: 'oldest', text: 'Oldest First', column: 'createdAt', direction: 'asc' },
          { key: 'alphabetical', text: 'A-Z by Name', column: 'name', direction: 'asc' },
          { key: 'priority_high', text: 'High Priority First', column: 'priority', direction: 'desc' },
          { key: 'recent_activity', text: 'Recent Activity', column: 'modifiedAt', direction: 'desc' }
        ]
      });
            
      this.setModel(this._oSortingModel, 'sorting');
    },
        
    // === COLUMN SORTING FUNCTIONALITY ===
        
    /**
         * Handles column header press for sorting
         */
    onColumnSort: function (oEvent) {
      const oColumn = oEvent.getSource();
      const sSortProperty = oColumn.getSortProperty();
            
      if (!sSortProperty) {
        MessageToast.show('This column cannot be sorted');
        return;
      }
            
      this._applySingleColumnSort(sSortProperty);
    },
        
    /**
         * Applies quick sort option
         */
    onQuickSort: function (oEvent) {
      const sSelectedKey = oEvent.getParameter('selectedItem').getKey();
      const aQuickSorts = this._oSortingModel.getProperty('/quickSorts');
      const oQuickSort = aQuickSorts.find(sort => sort.key === sSelectedKey);
            
      if (oQuickSort) {
        this._applySingleColumnSort(oQuickSort.column, oQuickSort.direction);
        MessageToast.show(`Quick sort applied: ${  oQuickSort.text}`);
      }
    },
        
    /**
         * Resets all sorting to default
         */
    onResetSorting: function () {
      this._oSortingModel.setProperty('/currentSort', {
        column: 'modifiedAt',
        direction: 'desc', 
        active: true
      });
            
      this._applySingleColumnSort('modifiedAt', 'desc');
      MessageToast.show('Sorting reset to default (Modified Date)');
    },
        
    /**
         * Applies single column sorting
         */
    _applySingleColumnSort: function (sColumn, sDirection) {
      const oTable = this.byId('projectsTable');
      const oBinding = oTable.getBinding('items');
            
      if (!oBinding) {
        return;
      }
            
      // Determine sort direction
      const oCurrentSort = this._oSortingModel.getProperty('/currentSort');
      let sNewDirection = sDirection;
            
      if (!sNewDirection) {
        if (oCurrentSort.column === sColumn) {
          sNewDirection = oCurrentSort.direction === 'asc' ? 'desc' : 'asc';
        } else {
          sNewDirection = this._getDefaultSortDirection(sColumn);
        }
      }
            
      // Create sorter with custom comparators if needed
      const oSorter = new sap.ui.model.Sorter(sColumn, sNewDirection === 'desc');
            
      if (sColumn === 'priority') {
        oSorter.fnCompare = this._priorityComparator.bind(this);
      } else if (sColumn === 'status') {
        oSorter.fnCompare = this._statusComparator.bind(this);
      }
            
      // Apply sorting
      oBinding.sort([oSorter]);
            
      // Update model
      this._oSortingModel.setProperty('/currentSort', {
        column: sColumn,
        direction: sNewDirection,
        active: true
      });
            
      // Update visual indicators
      this._updateSortIndicators();
            
      // Add to history
      this._addToSortHistory(sColumn, sNewDirection);
            
      const sColumnName = this._getColumnDisplayName(sColumn);
      const sDirectionText = sNewDirection === 'asc' ? 'ascending' : 'descending';
      MessageToast.show(`Sorted by ${  sColumnName  } (${  sDirectionText  })`);
    },
        
    /**
         * Custom comparator for priority sorting
         */
    _priorityComparator: function (a, b) {
      const priorityOrder = { 'CRITICAL': 4, 'HIGH': 3, 'MEDIUM': 2, 'LOW': 1 };
      const valueA = priorityOrder[a] || 0;
      const valueB = priorityOrder[b] || 0;
      return valueA - valueB;
    },
        
    /**
         * Custom comparator for status sorting
         */
    _statusComparator: function (a, b) {
      const statusOrder = {
        'ACTIVE': 7, 'DEPLOYED': 6, 'TESTING': 5, 'DRAFT': 4,
        'PAUSED': 3, 'COMPLETED': 2, 'ARCHIVED': 1, 'ERROR': 0
      };
      const valueA = statusOrder[a] || 0;
      const valueB = statusOrder[b] || 0;
      return valueA - valueB;
    },
        
    /**
         * Gets default sort direction for column
         */
    _getDefaultSortDirection: function (sColumn) {
      const defaultDirections = {
        'name': 'asc', 'status': 'desc', 'priority': 'desc',
        'createdAt': 'desc', 'modifiedAt': 'desc', 'lastDeployment': 'desc', 'createdBy': 'asc'
      };
      return defaultDirections[sColumn] || 'asc';
    },
        
    /**
         * Gets display name for column
         */
    _getColumnDisplayName: function (sColumn) {
      const columnNames = {
        'name': 'Project Name', 'status': 'Status', 'priority': 'Priority',
        'createdAt': 'Created Date', 'modifiedAt': 'Modified Date',
        'lastDeployment': 'Last Deployment', 'createdBy': 'Created By'
      };
      return columnNames[sColumn] || sColumn;
    },
        
    /**
         * Updates sort visual indicators
         */
    _updateSortIndicators: function () {
      const oTable = this.byId('projectsTable');
      const oCurrentSort = this._oSortingModel.getProperty('/currentSort');
            
      if (!oTable || !oCurrentSort) {
        return;
      }
            
      // Update sort indicators on column headers
      oTable.getColumns().forEach((oColumn, _iIndex) => {
        const sSortProperty = oColumn.getSortProperty();
                
        if (sSortProperty === oCurrentSort.column) {
          oColumn.setSortIndicator(oCurrentSort.direction === 'asc' ? 'Ascending' : 'Descending');
        } else {
          oColumn.setSortIndicator('None');
        }
      });
    },
        
    /**
         * Adds sort configuration to history
         */
    _addToSortHistory: function (sColumn, sDirection) {
      const aSortHistory = this._oSortingModel.getProperty('/sortHistory');
      const sDisplayName = this._getColumnDisplayName(sColumn);
            
      const oHistoryEntry = {
        name: `${sDisplayName  } (${  sDirection === 'asc' ? 'A-Z' : 'Z-A'  })`,
        column: sColumn,
        direction: sDirection,
        timestamp: new Date()
      };
            
      // Remove duplicate if exists
      const iExistingIndex = aSortHistory.findIndex(entry => 
        entry.column === sColumn && entry.direction === sDirection
      );
            
      if (iExistingIndex > -1) {
        aSortHistory.splice(iExistingIndex, 1);
      }
            
      // Add to beginning
      aSortHistory.unshift(oHistoryEntry);
            
      // Keep only last 5 entries
      if (aSortHistory.length > 5) {
        aSortHistory.splice(5);
      }
            
      this._oSortingModel.setProperty('/sortHistory', aSortHistory);
    },
        
    // === AGGREGATION FUNCTIONALITY ===
        
    /**
         * Initializes aggregation configuration
         */
    _initializeAggregations: function () {
      this._oAggregationModel = new JSONModel({
        enabled: false,
        currentAggregations: [],
        availableAggregations: [
          { 
            key: 'budget_sum', 
            text: 'Total Budget', 
            field: 'budget', 
            type: 'sum', 
            formatter: 'currency',
            icon: 'sap-icon://money-bills'
          },
          { 
            key: 'budget_avg', 
            text: 'Average Budget', 
            field: 'budget', 
            type: 'average', 
            formatter: 'currency',
            icon: 'sap-icon://average'
          },
          { 
            key: 'project_count', 
            text: 'Project Count', 
            field: '*', 
            type: 'count', 
            formatter: 'number',
            icon: 'sap-icon://sum'
          },
          { 
            key: 'agent_sum', 
            text: 'Total Agents', 
            field: 'agentCount', 
            type: 'sum', 
            formatter: 'number',
            icon: 'sap-icon://group'
          },
          { 
            key: 'success_rate', 
            text: 'Success Rate', 
            field: 'deploymentSuccessRate', 
            type: 'average', 
            formatter: 'percentage',
            icon: 'sap-icon://target-group'
          }
        ],
        aggregationSettings: {
          showInFooter: true,
          showInGroupHeaders: true,
          updateRealTime: true,
          highlightAggregations: true
        },
        aggregationResults: {},
        customAggregations: []
      });
            
      this.setModel(this._oAggregationModel, 'aggregation');
    },
        
    /**
         * Toggles aggregation display
         */
    onToggleAggregations: function () {
      const bEnabled = this._oAggregationModel.getProperty('/enabled');
      this._oAggregationModel.setProperty('/enabled', !bEnabled);
            
      if (!bEnabled) {
        this._applyAggregations();
        MessageToast.show('Aggregations enabled');
      } else {
        this._removeAggregations();
        MessageToast.show('Aggregations disabled');
      }
    },
        
    /**
         * Applies aggregations to table
         */
    _applyAggregations: function () {
      const _oTable = this.byId('projectsTable');
      const aCurrent = this._oAggregationModel.getProperty('/currentAggregations');
            
      // Use some default aggregations if none selected
      if (aCurrent.length === 0) {
        const aAvailable = this._oAggregationModel.getProperty('/availableAggregations');
        this._oAggregationModel.setProperty('/currentAggregations', [
          aAvailable[0], // Total Budget
          aAvailable[2]  // Project Count
        ]);
      }
            
      // Calculate aggregations
      this._calculateAggregations();
            
      // Add aggregation footer
      this._addAggregationFooter();
            
      // Update visual indicators
      this._updateAggregationIndicators();
    },
        
    /**
         * Removes aggregations from table
         */
    _removeAggregations: function () {
      this._removeAggregationFooter();
      this._oAggregationModel.setProperty('/aggregationResults', {});
      this._updateAggregationIndicators();
    },
        
    /**
         * Calculates aggregation values
         */
    _calculateAggregations: function () {
      const oTable = this.byId('projectsTable');
      const oBinding = oTable.getBinding('items');
            
      if (!oBinding) {
        return;
      }
            
      const aItems = oBinding.getContexts();
      const aCurrent = this._oAggregationModel.getProperty('/currentAggregations');
      const oResults = {};
            
      aCurrent.forEach(oAgg => {
        oResults[oAgg.key] = this._performAggregation(aItems, oAgg);
      });
            
      this._oAggregationModel.setProperty('/aggregationResults', oResults);
            
      // Update display
      if (this._oAggregationModel.getProperty('/enabled')) {
        this._updateAggregationDisplay();
      }
    },
        
    /**
         * Performs specific aggregation calculation
         */
    _performAggregation: function (aContexts, oAggregation) {
      let result = 0;
      let count = 0;
            
      aContexts.forEach(oContext => {
        // Skip group headers
        if (oContext.isGroupHeader && oContext.isGroupHeader()) {
          return;
        }
                
        const oData = oContext.getObject();
                
        switch (oAggregation.type) {
        case 'sum':
          if (oAggregation.field === 'agentCount') {
            result += (oData.agents ? oData.agents.length : 0);
          } else {
            result += parseFloat(oData[oAggregation.field]) || 0;
          }
          count++;
          break;
                        
        case 'average': {
          const value = parseFloat(oData[oAggregation.field]) || 0;
          result += value;
          count++;
          break;
        }
                        
        case 'count':
          count++;
          break;
        }
      });
            
      // Calculate final result
      if (oAggregation.type === 'average' && count > 0) {
        result = result / count;
      } else if (oAggregation.type === 'count') {
        result = count;
      }
            
      // Format result
      return this._formatAggregationResult(result, oAggregation);
    },
        
    /**
         * Formats aggregation result for display
         */
    _formatAggregationResult: function (value, oAggregation) {
      switch (oAggregation.formatter) {
      case 'currency': {
        const oCurrencyFormat = sap.ui.core.format.NumberFormat.getCurrencyInstance({
          showMeasure: true
        });
        return oCurrencyFormat.format(value, 'USD');
      }
                    
      case 'percentage':
        return `${Math.round(value)  }%`;
                    
      case 'number': {
        const oNumberFormat = sap.ui.core.format.NumberFormat.getIntegerInstance({
          groupingEnabled: true
        });
        return oNumberFormat.format(value);
      }
                    
      default:
        return value.toFixed(2);
      }
    },
        
    /**
         * Adds aggregation footer to table
         */
    _addAggregationFooter: function () {
      const oTable = this.byId('projectsTable');
            
      // Remove existing footer
      this._removeAggregationFooter();
            
      // Create footer toolbar
      const oFooterBar = new sap.m.OverflowToolbar({
        id: this.createId('aggregationFooter'),
        design: 'Solid',
        style: 'Clear'
      });
            
      // Add title
      oFooterBar.addContent(new sap.m.Label({
        text: 'Aggregations:',
        design: 'Bold'
      }));
            
      oFooterBar.addContent(new sap.m.ToolbarSeparator());
            
      // Add aggregation results
      const oResults = this._oAggregationModel.getProperty('/aggregationResults');
      const aCurrent = this._oAggregationModel.getProperty('/currentAggregations');
            
      aCurrent.forEach((oAgg, index) => {
        if (index > 0) {
          oFooterBar.addContent(new sap.m.ToolbarSeparator());
        }
                
        // Add icon
        oFooterBar.addContent(new sap.ui.core.Icon({
          src: oAgg.icon,
          color: '#0854a0'
        }));
                
        // Add label and value
        oFooterBar.addContent(new sap.m.Label({
          text: `${oAgg.text  }:`,
          design: 'Bold'
        }));
                
        oFooterBar.addContent(new sap.m.Text({
          text: oResults[oAgg.key] || '0'
        }));
      });
            
      oTable.setFooter(oFooterBar);
    },
        
    /**
         * Removes aggregation footer from table
         */
    _removeAggregationFooter: function () {
      const oTable = this.byId('projectsTable');
      const oFooter = this.byId('aggregationFooter');
            
      if (oFooter) {
        oTable.setFooter(null);
        oFooter.destroy();
      }
    },
        
    /**
         * Updates aggregation display
         */
    _updateAggregationDisplay: function () {
      if (this._oAggregationModel.getProperty('/aggregationSettings/showInFooter')) {
        this._addAggregationFooter();
      }
    },
        
    /**
         * Updates aggregation visual indicators
         */
    _updateAggregationIndicators: function () {
      const bEnabled = this._oAggregationModel.getProperty('/enabled');
      const oAggButton = this.byId('aggregationButton');
            
      if (oAggButton) {
        oAggButton.setType(bEnabled ? 'Emphasized' : 'Default');
        const aCurrent = this._oAggregationModel.getProperty('/currentAggregations');
        oAggButton.setTooltip(bEnabled ? 
          `Aggregations active (${  aCurrent.length  })` : 
          'Enable aggregations');
      }
    },
        
    /* =========================================================== */
    /* Chart Toggle Functionality                                  */
    /* =========================================================== */
        
    /**
         * Handles chart view toggle
         */
    onToggleChartView: function () {
      const oViewModel = this.getModel('view');
      const bEnabled = !oViewModel.getProperty('/chartView/enabled');
            
      oViewModel.setProperty('/chartView/enabled', bEnabled);
            
      // Initialize charts on first toggle
      if (bEnabled && !this._bChartsInitialized) {
        this._initializeCharts();
        this._bChartsInitialized = true;
      }
            
      // Update chart data
      if (bEnabled) {
        this._updateChartData();
      }
            
      // Update button state
      const oChartButton = this.byId('chartToggleButton');
      if (oChartButton) {
        oChartButton.setType(bEnabled ? 'Emphasized' : 'Default');
        oChartButton.setTooltip(bEnabled ? 
          'Hide chart visualization' : 
          'Show chart visualization');
      }
            
      // Show/hide table based on chart view
      const oSmartTable = this.byId('projectsSmartTable');
      if (oSmartTable) {
        oSmartTable.setVisible(!bEnabled);
      }
            
      MessageToast.show(bEnabled ? 
        'Chart view enabled' : 
        'Table view restored');
    },
        
    /**
         * Initialize chart libraries and containers
         */
    _initializeCharts: function () {
      const oViewModel = this.getModel('view');
            
      // Set default chart configuration
      oViewModel.setProperty('/chartView', {
        enabled: false,
        selectedChart: 'statusChart',
        chartType: 'pie',
        charts: {
          status: null,
          priority: null,
          timeline: null,
          department: null
        },
        data: {
          status: [],
          priority: [],
          timeline: [],
          department: []
        }
      });
            
      // Load chart library
      sap.ui.require(['sap/viz/ui5/controls/VizFrame', 
        'sap/viz/ui5/data/FlattenedDataset',
        'sap/viz/ui5/controls/common/feeds/FeedItem'], 
      this._createCharts.bind(this));
    },
        
    /**
         * Create chart instances
         */
    _createCharts: function (VizFrame, FlattenedDataset, FeedItem) {
      this._VizFrame = VizFrame;
      this._FlattenedDataset = FlattenedDataset;
      this._FeedItem = FeedItem;
            
      // Create status distribution chart
      this._createStatusChart();
            
      // Create priority analysis chart
      this._createPriorityChart();
            
      // Create timeline chart
      this._createTimelineChart();
            
      // Create department distribution chart
      this._createDepartmentChart();
    },
        
    /**
         * Create status distribution chart
         */
    _createStatusChart: function () {
      const oViewModel = this.getModel('view');
      const oContainer = this.byId('statusChartContainer');
            
      if (!oContainer) {
        return;
      }
            
      const oVizFrame = new this._VizFrame({
        height: '400px',
        width: '100%',
        vizType: 'pie'
      });
            
      // Configure chart properties
      oVizFrame.setVizProperties({
        title: {
          visible: false
        },
        plotArea: {
          dataLabel: {
            visible: true,
            showTotal: true
          },
          colorPalette: ['#5899DA', '#E8743B', '#19A979', 
            '#ED4A7B', '#945ECF', '#13A4B4']
        },
        legend: {
          visible: true,
          title: {
            visible: false
          }
        }
      });
            
      // Create dataset
      const oDataset = new this._FlattenedDataset({
        dimensions: [{
          name: 'Status',
          value: '{status}'
        }],
        measures: [{
          name: 'Count',
          value: '{count}'
        }],
        data: {
          path: '/chartView/data/status'
        }
      });
            
      oVizFrame.setDataset(oDataset);
            
      // Add feeds
      const oFeedValueAxis = new this._FeedItem({
        uid: 'size',
        type: 'Measure',
        values: ['Count']
      });
      const oFeedCategoryAxis = new this._FeedItem({
        uid: 'color',
        type: 'Dimension',
        values: ['Status']
      });
            
      oVizFrame.addFeed(oFeedValueAxis);
      oVizFrame.addFeed(oFeedCategoryAxis);
            
      // Add to container
      oContainer.removeAllItems();
      oContainer.addItem(oVizFrame);
            
      // Store reference
      oViewModel.setProperty('/chartView/charts/status', oVizFrame);
    },
        
    /**
         * Create priority analysis chart
         */
    _createPriorityChart: function () {
      const oViewModel = this.getModel('view');
      const oContainer = this.byId('priorityChartContainer');
            
      if (!oContainer) {
        return;
      }
            
      const oVizFrame = new this._VizFrame({
        height: '400px',
        width: '100%',
        vizType: 'bar'
      });
            
      oVizFrame.setVizProperties({
        title: {
          visible: false
        },
        plotArea: {
          dataLabel: {
            visible: true
          },
          colorPalette: ['#dc3545', '#ffc107', '#28a745', '#17a2b8']
        },
        legend: {
          visible: false
        }
      });
            
      const oDataset = new this._FlattenedDataset({
        dimensions: [{
          name: 'Priority',
          value: '{priority}'
        }],
        measures: [{
          name: 'Projects',
          value: '{count}'
        }],
        data: {
          path: '/chartView/data/priority'
        }
      });
            
      oVizFrame.setDataset(oDataset);
            
      const oFeedValueAxis = new this._FeedItem({
        uid: 'valueAxis',
        type: 'Measure',
        values: ['Projects']
      });
      const oFeedCategoryAxis = new this._FeedItem({
        uid: 'categoryAxis',
        type: 'Dimension',
        values: ['Priority']
      });
            
      oVizFrame.addFeed(oFeedValueAxis);
      oVizFrame.addFeed(oFeedCategoryAxis);
            
      oContainer.removeAllItems();
      oContainer.addItem(oVizFrame);
            
      oViewModel.setProperty('/chartView/charts/priority', oVizFrame);
    },
        
    /**
         * Create timeline chart
         */
    _createTimelineChart: function () {
      const oViewModel = this.getModel('view');
      const oContainer = this.byId('timelineChartContainer');
            
      if (!oContainer) {
        return;
      }
            
      const oVizFrame = new this._VizFrame({
        height: '400px',
        width: '100%',
        vizType: 'line'
      });
            
      oVizFrame.setVizProperties({
        title: {
          visible: false
        },
        plotArea: {
          window: {
            start: 'firstDataPoint',
            end: 'lastDataPoint'
          },
          dataLabel: {
            visible: false
          }
        },
        legend: {
          visible: false
        },
        timeAxis: {
          title: {
            visible: false
          }
        },
        valueAxis: {
          title: {
            visible: false
          }
        }
      });
            
      const oDataset = new this._FlattenedDataset({
        dimensions: [{
          name: 'Date',
          value: '{date}',
          dataType: 'date'
        }],
        measures: [{
          name: 'Projects Created',
          value: '{count}'
        }],
        data: {
          path: '/chartView/data/timeline'
        }
      });
            
      oVizFrame.setDataset(oDataset);
            
      const oFeedValueAxis = new this._FeedItem({
        uid: 'valueAxis',
        type: 'Measure',
        values: ['Projects Created']
      });
      const oFeedTimeAxis = new this._FeedItem({
        uid: 'timeAxis',
        type: 'Dimension',
        values: ['Date']
      });
            
      oVizFrame.addFeed(oFeedValueAxis);
      oVizFrame.addFeed(oFeedTimeAxis);
            
      oContainer.removeAllItems();
      oContainer.addItem(oVizFrame);
            
      oViewModel.setProperty('/chartView/charts/timeline', oVizFrame);
    },
        
    /**
         * Create department distribution chart
         */
    _createDepartmentChart: function () {
      const oViewModel = this.getModel('view');
      const oContainer = this.byId('departmentChartContainer');
            
      if (!oContainer) {
        return;
      }
            
      const oVizFrame = new this._VizFrame({
        height: '400px',
        width: '100%',
        vizType: 'donut'
      });
            
      oVizFrame.setVizProperties({
        title: {
          visible: false
        },
        plotArea: {
          dataLabel: {
            visible: true,
            type: 'percentage'
          },
          colorPalette: ['#5899DA', '#E8743B', '#19A979', 
            '#ED4A7B', '#945ECF', '#13A4B4',
            '#FF6B6B', '#4ECDC4', '#45B7D1']
        },
        legend: {
          visible: true,
          title: {
            visible: false
          }
        }
      });
            
      const oDataset = new this._FlattenedDataset({
        dimensions: [{
          name: 'Department',
          value: '{department}'
        }],
        measures: [{
          name: 'Projects',
          value: '{count}'
        }],
        data: {
          path: '/chartView/data/department'
        }
      });
            
      oVizFrame.setDataset(oDataset);
            
      const oFeedValueAxis = new this._FeedItem({
        uid: 'size',
        type: 'Measure',
        values: ['Projects']
      });
      const oFeedCategoryAxis = new this._FeedItem({
        uid: 'color',
        type: 'Dimension',
        values: ['Department']
      });
            
      oVizFrame.addFeed(oFeedValueAxis);
      oVizFrame.addFeed(oFeedCategoryAxis);
            
      oContainer.removeAllItems();
      oContainer.addItem(oVizFrame);
            
      oViewModel.setProperty('/chartView/charts/department', oVizFrame);
    },
        
    /**
         * Update chart data based on current table data
         */
    _updateChartData: function () {
      const oTable = this.byId('projectsTable');
      if (!oTable) {
        return;
      }
            
      const aItems = oTable.getItems();
      const oViewModel = this.getModel('view');
            
      // Process data for different charts
      const oStatusData = {};
      const oPriorityData = {};
      const oDepartmentData = {};
      const oTimelineData = {};
            
      aItems.forEach(oItem => {
        const oContext = oItem.getBindingContext();
        if (!oContext) {
          return;
        }
                
        // Status distribution
        const sStatus = oContext.getProperty('status');
        oStatusData[sStatus] = (oStatusData[sStatus] || 0) + 1;
                
        // Priority distribution
        const sPriority = oContext.getProperty('priority');
        oPriorityData[sPriority] = (oPriorityData[sPriority] || 0) + 1;
                
        // Department distribution
        const sDepartment = oContext.getProperty('department') || 'Unassigned';
        oDepartmentData[sDepartment] = (oDepartmentData[sDepartment] || 0) + 1;
                
        // Timeline data (by month)
        const dCreated = oContext.getProperty('createdDate');
        if (dCreated) {
          const sMonth = new Date(dCreated).toISOString().substring(0, 7);
          oTimelineData[sMonth] = (oTimelineData[sMonth] || 0) + 1;
        }
      });
            
      // Convert to array format for charts
      const aStatusData = Object.keys(oStatusData).map(key => ({
        status: key,
        count: oStatusData[key]
      }));
            
      const aPriorityData = Object.keys(oPriorityData).map(key => ({
        priority: key,
        count: oPriorityData[key]
      }));
            
      const aDepartmentData = Object.keys(oDepartmentData).map(key => ({
        department: key,
        count: oDepartmentData[key]
      }));
            
      const aTimelineData = Object.keys(oTimelineData).sort().map(key => ({
        date: `${key  }-01`,
        count: oTimelineData[key]
      }));
            
      // Update model
      oViewModel.setProperty('/chartView/data/status', aStatusData);
      oViewModel.setProperty('/chartView/data/priority', aPriorityData);
      oViewModel.setProperty('/chartView/data/department', aDepartmentData);
      oViewModel.setProperty('/chartView/data/timeline', aTimelineData);
    },
        
    /**
         * Handle chart type change
         */
    onChartTypeChange: function (oEvent) {
      const sChartType = oEvent.getParameter('selectedItem').getKey();
      const oViewModel = this.getModel('view');
      const sSelectedChart = oViewModel.getProperty('/chartView/selectedChart');
            
      // Get the current chart
      const oVizFrame = oViewModel.getProperty(`/chartView/charts/${  
        sSelectedChart.replace('Chart', '')}`);
            
      if (oVizFrame) {
        // Update chart type
        oVizFrame.setVizType(sChartType);
                
        // Update properties based on type
        this._updateChartProperties(oVizFrame, sChartType);
      }
    },
        
    /**
         * Update chart properties based on type
         */
    _updateChartProperties: function (oVizFrame, sChartType) {
      const oProperties = {
        title: { visible: false },
        plotArea: {
          dataLabel: {
            visible: true
          }
        },
        legend: {
          visible: ['pie', 'donut'].includes(sChartType)
        }
      };
            
      oVizFrame.setVizProperties(oProperties);
    },
        
    /**
         * Refresh chart data
         */
    onRefreshCharts: function () {
      this._updateChartData();
      MessageToast.show('Charts refreshed');
    },
        
    /**
         * Export current chart
         */
    onExportChart: function () {
      const oViewModel = this.getModel('view');
      const sSelectedChart = oViewModel.getProperty('/chartView/selectedChart');
      const oVizFrame = oViewModel.getProperty(`/chartView/charts/${  
        sSelectedChart.replace('Chart', '')}`);
            
      if (oVizFrame) {
        // Get chart as SVG
        const sSvg = oVizFrame.exportToSVGString({
          width: 800,
          height: 600
        });
                
        // Create download link
        const blob = new Blob([sSvg], { type: 'image/svg+xml' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `${sSelectedChart  }_${  new Date().toISOString()  }.svg`;
        a.click();
        URL.revokeObjectURL(url);
                
        MessageToast.show('Chart exported successfully');
      }
    },
        
    /* =========================================================== */
    /* Print Preview Functionality                                 */
    /* =========================================================== */
        
    /**
         * Opens print preview dialog
         */
    onPrintPreview: function () {
      if (!this._oPrintPreviewDialog) {
        Fragment.load({
          id: this.getView().getId(),
          name: 'sap.a2a.view.fragments.PrintPreviewDialog',
          controller: this
        }).then((oDialog) => {
          this._oPrintPreviewDialog = oDialog;
          this.getView().addDependent(this._oPrintPreviewDialog);
                    
          // Initialize print preview model
          this._initializePrintPreview();
                    
          // Open dialog
          this._oPrintPreviewDialog.open();
        });
      } else {
        // Update preview content
        this._updatePrintPreview();
        this._oPrintPreviewDialog.open();
      }
    },
        
    /**
         * Initialize print preview model and settings
         */
    _initializePrintPreview: function () {
      const oPrintModel = new JSONModel({
        settings: {
          paperSize: 'A4',
          orientation: 'portrait',
          margins: 'normal',
          includeHeader: true,
          includeKPIs: true,
          includeFilters: true,
          includePageNumbers: true,
          includeTimestamp: true,
          rangeType: 0, // 0=All, 1=Current, 2=Custom
          customRange: '',
          columns: {
            name: true,
            status: true,
            priority: true,
            agents: true,
            lastDeployment: true,
            createdBy: true
          }
        },
        previewScale: '100',
        currentPage: 1,
        totalPages: 1,
        pageSize: 20,
        previewContent: []
      });
            
      this.setModel(oPrintModel, 'printPreview');
            
      // Generate initial preview
      this._updatePrintPreview();
    },
        
    /**
         * Update print preview content based on settings
         */
    _updatePrintPreview: function () {
      const oPrintModel = this.getModel('printPreview');
      const oSettings = oPrintModel.getProperty('/settings');
      const oTable = this.byId('projectsTable');
            
      if (!oTable) {
        return;
      }
            
      // Get visible items
      const aItems = oTable.getItems();
      const iPageSize = this._calculatePageSize(oSettings);
      const iTotalPages = Math.ceil(aItems.length / iPageSize);
            
      // Update page info
      oPrintModel.setProperty('/pageSize', iPageSize);
      oPrintModel.setProperty('/totalPages', iTotalPages);
      oPrintModel.setProperty('/currentPage', 1);
            
      // Generate preview content
      this._generatePreviewContent();
    },
        
    /**
         * Calculate page size based on paper and orientation
         */
    _calculatePageSize: function (oSettings) {
      const mPageSizes = {
        'A4': { portrait: 25, landscape: 15 },
        'Letter': { portrait: 22, landscape: 14 },
        'Legal': { portrait: 28, landscape: 18 },
        'A3': { portrait: 35, landscape: 22 }
      };
            
      const sPaper = oSettings.paperSize;
      const sOrientation = oSettings.orientation;
            
      return mPageSizes[sPaper][sOrientation] || 20;
    },
        
    /**
         * Generate preview content for current page
         */
    _generatePreviewContent: function () {
      const oPrintModel = this.getModel('printPreview');
      const oSettings = oPrintModel.getProperty('/settings');
      const iCurrentPage = oPrintModel.getProperty('/currentPage');
      const iPageSize = oPrintModel.getProperty('/pageSize');
      const oContainer = this.byId('printPreviewContainer');
            
      if (!oContainer) {
        return;
      }
            
      // Clear existing content
      oContainer.removeAllItems();
            
      // Add header if enabled
      if (oSettings.includeHeader) {
        this._addPrintHeader(oContainer);
      }
            
      // Add KPIs if enabled
      if (oSettings.includeKPIs) {
        this._addPrintKPIs(oContainer);
      }
            
      // Add filters summary if enabled
      if (oSettings.includeFilters) {
        this._addPrintFilters(oContainer);
      }
            
      // Add table content
      this._addPrintTable(oContainer, iCurrentPage, iPageSize);
            
      // Add footer if page numbers or timestamp enabled
      if (oSettings.includePageNumbers || oSettings.includeTimestamp) {
        this._addPrintFooter(oContainer, iCurrentPage);
      }
    },
        
    /**
         * Add print header to preview
         */
    _addPrintHeader: function (oContainer) {
      const oHeader = new VBox({
        class: 'sapUiMediumMarginBottom a2a-print-header'
      });
            
      oHeader.addItem(new Title({
        text: 'Projects List Report',
        level: 'H1',
        class: 'sapUiSmallMarginBottom'
      }));
            
      oHeader.addItem(new Text({
        text: 'A2A Agent Development Portal',
        class: 'sapMTextColorSecondary'
      }));
            
      oHeader.addItem(new sap.m.Separator({
        class: 'sapUiSmallMarginTop'
      }));
            
      oContainer.addItem(oHeader);
    },
        
    /**
         * Add KPI cards to print preview
         */
    _addPrintKPIs: function (oContainer) {
      const oViewModel = this.getModel('listReport');
      const oKPIs = oViewModel.getProperty('/kpi');
            
      const oKPIBox = new HBox({
        class: 'sapUiMediumMarginBottom',
        justifyContent: 'SpaceAround'
      });
            
      // Total Projects
      oKPIBox.addItem(new VBox({
        alignItems: 'Center',
        class: 'sapUiSmallMargin'
      }).addItem(new Text({
        text: 'Total Projects',
        class: 'sapMTextBold'
      })).addItem(new Text({
        text: oKPIs.totalProjects.toString(),
        class: 'sapMTextLarge'
      })));
            
      // Active Projects
      oKPIBox.addItem(new VBox({
        alignItems: 'Center',
        class: 'sapUiSmallMargin'
      }).addItem(new Text({
        text: 'Active Projects',
        class: 'sapMTextBold'
      })).addItem(new Text({
        text: oKPIs.activeProjects.toString(),
        class: 'sapMTextLarge'
      })));
            
      // Completed Projects
      oKPIBox.addItem(new VBox({
        alignItems: 'Center',
        class: 'sapUiSmallMargin'
      }).addItem(new Text({
        text: 'Completed Projects',
        class: 'sapMTextBold'
      })).addItem(new Text({
        text: oKPIs.completedProjects.toString(),
        class: 'sapMTextLarge'
      })));
            
      // Success Rate
      oKPIBox.addItem(new VBox({
        alignItems: 'Center',
        class: 'sapUiSmallMargin'
      }).addItem(new Text({
        text: 'Success Rate',
        class: 'sapMTextBold'
      })).addItem(new Text({
        text: `${oKPIs.deploymentSuccessRate  }%`,
        class: 'sapMTextLarge'
      })));
            
      oContainer.addItem(oKPIBox);
      oContainer.addItem(new sap.m.Separator({
        class: 'sapUiSmallMarginBottom'
      }));
    },
        
    /**
         * Add filters summary to print preview
         */
    _addPrintFilters: function (oContainer) {
      const oFilterBar = this.byId('listReportFilter');
      if (!oFilterBar) {
        return;
      }
            
      const aFilters = oFilterBar.getFilterConditions();
      const aActiveFilters = [];
            
      // Extract active filters
      Object.keys(aFilters).forEach(sKey => {
        const oFilter = aFilters[sKey];
        if (oFilter && oFilter.length > 0) {
          aActiveFilters.push({
            field: sKey,
            values: oFilter.map(f => f.values[0]).join(', ')
          });
        }
      });
            
      if (aActiveFilters.length > 0) {
        const oFiltersBox = new VBox({
          class: 'sapUiMediumMarginBottom'
        });
                
        oFiltersBox.addItem(new Text({
          text: 'Active Filters:',
          class: 'sapMTextBold sapUiSmallMarginBottom'
        }));
                
        aActiveFilters.forEach(oFilter => {
          oFiltersBox.addItem(new Text({
            text: `${oFilter.field  }: ${  oFilter.values}`,
            class: 'sapUiTinyMarginBottom'
          }));
        });
                
        oContainer.addItem(oFiltersBox);
        oContainer.addItem(new sap.m.Separator({
          class: 'sapUiSmallMarginBottom'
        }));
      }
    },
        
    /**
         * Add table content to print preview
         */
    _addPrintTable: function (oContainer, iCurrentPage, iPageSize) {
      const oPrintModel = this.getModel('printPreview');
      const oSettings = oPrintModel.getProperty('/settings');
      const oTable = this.byId('projectsTable');
            
      if (!oTable) {
        return;
      }
            
      // Create print table
      const oPrintTable = new Table({
        showSeparators: 'All',
        fixedLayout: false
      });
            
      // Add columns based on settings
      if (oSettings.columns.name !== false) {
        oPrintTable.addColumn(new Column({
          header: new Text({ text: 'Project Name' }),
          width: '25%'
        }));
      }
      if (oSettings.columns.status) {
        oPrintTable.addColumn(new Column({
          header: new Text({ text: 'Status' }),
          width: '15%'
        }));
      }
      if (oSettings.columns.priority) {
        oPrintTable.addColumn(new Column({
          header: new Text({ text: 'Priority' }),
          width: '15%'
        }));
      }
      if (oSettings.columns.agents) {
        oPrintTable.addColumn(new Column({
          header: new Text({ text: 'Agents' }),
          width: '15%'
        }));
      }
      if (oSettings.columns.lastDeployment) {
        oPrintTable.addColumn(new Column({
          header: new Text({ text: 'Last Deployment' }),
          width: '15%'
        }));
      }
      if (oSettings.columns.createdBy) {
        oPrintTable.addColumn(new Column({
          header: new Text({ text: 'Created By' }),
          width: '15%'
        }));
      }
            
      // Get items for current page
      const aAllItems = oTable.getItems();
      const iStartIndex = (iCurrentPage - 1) * iPageSize;
      const iEndIndex = Math.min(iStartIndex + iPageSize, aAllItems.length);
      const aPageItems = aAllItems.slice(iStartIndex, iEndIndex);
            
      // Add items to print table
      aPageItems.forEach(oItem => {
        const oContext = oItem.getBindingContext();
        if (!oContext) {
          return;
        }
                
        const oPrintItem = new ColumnListItem();
                
        if (oSettings.columns.name !== false) {
          oPrintItem.addCell(new VBox({
            items: [
              new Text({ 
                text: oContext.getProperty('name'),
                class: 'sapMTextBold'
              }),
              new Text({ 
                text: oContext.getProperty('description'),
                class: 'sapUiTinyText sapMTextColorSecondary',
                maxLines: 2
              })
            ]
          }));
        }
        if (oSettings.columns.status) {
          oPrintItem.addCell(new Text({
            text: oContext.getProperty('status')
          }));
        }
        if (oSettings.columns.priority) {
          oPrintItem.addCell(new Text({
            text: oContext.getProperty('priority')
          }));
        }
        if (oSettings.columns.agents) {
          const aAgents = oContext.getProperty('agents') || [];
          oPrintItem.addCell(new Text({
            text: `${aAgents.length  } agents`
          }));
        }
        if (oSettings.columns.lastDeployment) {
          oPrintItem.addCell(new Text({
            text: this.formatRelativeTime(oContext.getProperty('lastDeployment'))
          }));
        }
        if (oSettings.columns.createdBy) {
          oPrintItem.addCell(new Text({
            text: oContext.getProperty('createdBy')
          }));
        }
                
        oPrintTable.addItem(oPrintItem);
      });
            
      oContainer.addItem(oPrintTable);
    },
        
    /**
         * Add footer to print preview
         */
    _addPrintFooter: function (oContainer, iCurrentPage) {
      const oPrintModel = this.getModel('printPreview');
      const oSettings = oPrintModel.getProperty('/settings');
      const iTotalPages = oPrintModel.getProperty('/totalPages');
            
      const oFooter = new HBox({
        justifyContent: 'SpaceBetween',
        class: 'sapUiMediumMarginTop a2a-print-footer'
      });
            
      // Add timestamp if enabled
      if (oSettings.includeTimestamp) {
        oFooter.addItem(new Text({
          text: `Printed: ${  new Date().toLocaleString()}`,
          class: 'sapMTextColorSecondary'
        }));
      } else {
        oFooter.addItem(new Text({ text: '' })); // Spacer
      }
            
      // Add page numbers if enabled
      if (oSettings.includePageNumbers) {
        oFooter.addItem(new Text({
          text: `Page ${  iCurrentPage  } of ${  iTotalPages}`,
          class: 'sapMTextColorSecondary'
        }));
      }
            
      oContainer.addItem(new sap.m.Separator({
        class: 'sapUiSmallMarginTop'
      }));
      oContainer.addItem(oFooter);
    },
        
    /**
         * Handle print setting changes
         */
    onPrintSettingChange: function () {
      this._updatePrintPreview();
      this._generatePreviewContent();
    },
        
    /**
         * Handle orientation change
         */
    onOrientationChange: function (oEvent) {
      const iSelectedIndex = oEvent.getParameter('selectedIndex');
      const oPrintModel = this.getModel('printPreview');
      oPrintModel.setProperty('/settings/orientation', 
        iSelectedIndex === 0 ? 'portrait' : 'landscape');
      this._updatePrintPreview();
      this._generatePreviewContent();
    },
        
    /**
         * Handle range type change
         */
    onRangeTypeChange: function (oEvent) {
      const iSelectedIndex = oEvent.getParameter('selectedIndex');
      const oPrintModel = this.getModel('printPreview');
      oPrintModel.setProperty('/settings/rangeType', iSelectedIndex);
    },
        
    /**
         * Handle preview scale change
         */
    onPreviewScaleChange: function (oEvent) {
      const sScale = oEvent.getParameter('item').getKey();
      const oPrintModel = this.getModel('printPreview');
      oPrintModel.setProperty('/previewScale', sScale);
    },
        
    /**
         * Zoom in preview
         */
    onZoomIn: function () {
      const oPrintModel = this.getModel('printPreview');
      let iScale = parseInt(oPrintModel.getProperty('/previewScale'));
      if (iScale < 200) {
        iScale += 25;
        oPrintModel.setProperty('/previewScale', iScale.toString());
      }
    },
        
    /**
         * Zoom out preview
         */
    onZoomOut: function () {
      const oPrintModel = this.getModel('printPreview');
      let iScale = parseInt(oPrintModel.getProperty('/previewScale'));
      if (iScale > 25) {
        iScale -= 25;
        oPrintModel.setProperty('/previewScale', iScale.toString());
      }
    },
        
    /**
         * Fit preview to screen
         */
    onFitToScreen: function () {
      const oPrintModel = this.getModel('printPreview');
      oPrintModel.setProperty('/previewScale', '75');
    },
        
    /**
         * Navigate to first page
         */
    onFirstPage: function () {
      const oPrintModel = this.getModel('printPreview');
      oPrintModel.setProperty('/currentPage', 1);
      this._generatePreviewContent();
    },
        
    /**
         * Navigate to previous page
         */
    onPreviousPage: function () {
      const oPrintModel = this.getModel('printPreview');
      const iCurrentPage = oPrintModel.getProperty('/currentPage');
      if (iCurrentPage > 1) {
        oPrintModel.setProperty('/currentPage', iCurrentPage - 1);
        this._generatePreviewContent();
      }
    },
        
    /**
         * Navigate to next page
         */
    onNextPage: function () {
      const oPrintModel = this.getModel('printPreview');
      const iCurrentPage = oPrintModel.getProperty('/currentPage');
      const iTotalPages = oPrintModel.getProperty('/totalPages');
      if (iCurrentPage < iTotalPages) {
        oPrintModel.setProperty('/currentPage', iCurrentPage + 1);
        this._generatePreviewContent();
      }
    },
        
    /**
         * Navigate to last page
         */
    onLastPage: function () {
      const oPrintModel = this.getModel('printPreview');
      const iTotalPages = oPrintModel.getProperty('/totalPages');
      oPrintModel.setProperty('/currentPage', iTotalPages);
      this._generatePreviewContent();
    },
        
    /**
         * Handle print action
         */
    onPrint: function () {
      // Create print-friendly content
      const oPrintWindow = window.open('', '_blank');
      const oPrintModel = this.getModel('printPreview');
      const oSettings = oPrintModel.getProperty('/settings');
            
      // Generate print HTML
      const sPrintHTML = this._generatePrintHTML(oSettings);
            
      // Write to print window
      oPrintWindow.document.write(sPrintHTML);
      oPrintWindow.document.close();
            
      // Trigger print dialog
      setTimeout(() => {
        oPrintWindow.print();
        oPrintWindow.close();
      }, 500);
            
      MessageToast.show('Opening print dialog...');
    },
        
    /**
         * Generate HTML for printing
         */
    _generatePrintHTML: function (oSettings) {
      let sHTML = `<!DOCTYPE html>
<html>
<head>
    <title>Projects List Report</title>
    <style>
        @page {
            size: ${oSettings.paperSize} ${oSettings.orientation};
            margin: ${oSettings.margins === 'normal' ? '1in' : 
    oSettings.margins === 'narrow' ? '0.5in' : 
      oSettings.margins === 'wide' ? '1.5in' : '0'};
        }
        body {
            font-family: Arial, sans-serif;
            font-size: 10pt;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f5f5f5;
            font-weight: bold;
        }
        .header {
            text-align: center;
            margin-bottom: 20px;
        }
        .footer {
            position: fixed;
            bottom: 0;
            width: 100%;
            text-align: center;
            font-size: 9pt;
            color: #666;
        }
        @media print {
            .pagebreak {
                page-break-after: always;
            }
        }
    </style>
</head>
<body>`;
            
      // Add content based on settings
      if (oSettings.includeHeader) {
        sHTML += `
    <div class="header">
        <h1>Projects List Report</h1>
        <p>A2A Agent Development Portal</p>
    </div>`;
      }
            
      // Add table content
      sHTML += this._generatePrintTableHTML(oSettings);
            
      // Add footer
      if (oSettings.includeTimestamp || oSettings.includePageNumbers) {
        sHTML += `
    <div class="footer">`;
        if (oSettings.includeTimestamp) {
          sHTML += `<span>Printed: ${new Date().toLocaleString()}</span>`;
        }
        sHTML += `
    </div>`;
      }
            
      sHTML += `
</body>
</html>`;
            
      return sHTML;
    },
        
    /**
         * Generate table HTML for printing
         */
    _generatePrintTableHTML: function (oSettings) {
      const oTable = this.byId('projectsTable');
      if (!oTable) {
        return '';
      }
            
      let sHTML = '<table><thead><tr>';
            
      // Add headers
      if (oSettings.columns.name !== false) {
        sHTML += '<th>Project Name</th>';
      }
      if (oSettings.columns.status) {
        sHTML += '<th>Status</th>';
      }
      if (oSettings.columns.priority) {
        sHTML += '<th>Priority</th>';
      }
      if (oSettings.columns.agents) {
        sHTML += '<th>Agents</th>';
      }
      if (oSettings.columns.lastDeployment) {
        sHTML += '<th>Last Deployment</th>';
      }
      if (oSettings.columns.createdBy) {
        sHTML += '<th>Created By</th>';
      }
            
      sHTML += '</tr></thead><tbody>';
            
      // Add rows
      const aItems = oTable.getItems();
      aItems.forEach(oItem => {
        const oContext = oItem.getBindingContext();
        if (!oContext) {
          return;
        }
                
        sHTML += '<tr>';
        if (oSettings.columns.name !== false) {
          sHTML += `<td>${oContext.getProperty('name')}<br>
                             <small>${oContext.getProperty('description') || ''}</small></td>`;
        }
        if (oSettings.columns.status) {
          sHTML += `<td>${oContext.getProperty('status')}</td>`;
        }
        if (oSettings.columns.priority) {
          sHTML += `<td>${oContext.getProperty('priority')}</td>`;
        }
        if (oSettings.columns.agents) {
          const aAgents = oContext.getProperty('agents') || [];
          sHTML += `<td>${aAgents.length} agents</td>`;
        }
        if (oSettings.columns.lastDeployment) {
          sHTML += `<td>${this.formatRelativeTime(oContext.getProperty('lastDeployment'))}</td>`;
        }
        if (oSettings.columns.createdBy) {
          sHTML += `<td>${oContext.getProperty('createdBy')}</td>`;
        }
        sHTML += '</tr>';
      });
            
      sHTML += '</tbody></table>';
      return sHTML;
    },
        
    /**
         * Export to PDF
         */
    onExportPDF: function () {
      // This would typically use a library like jsPDF or server-side PDF generation
      MessageBox.information('PDF export would be implemented using a PDF generation library or server-side service.');
    },
        
    /**
         * Open page setup dialog
         */
    onPageSetup: function () {
      const _oPrintModel = this.getModel('printPreview');
      const oPanel = this.byId('printSettingsPanel');
      if (oPanel) {
        oPanel.setExpanded(!oPanel.getExpanded());
      }
    },
        
    /**
         * Close print preview dialog
         */
    onClosePrintPreview: function () {
      if (this._oPrintPreviewDialog) {
        this._oPrintPreviewDialog.close();
      }
    },
        
    /* =========================================================== */
    /* Export Functionality                                        */
    /* =========================================================== */
        
    /**
         * Export table data to Excel
         */
    onExportExcel: function () {
      const _oTable = this.byId('projectsTable');
      const aColumns = this._getExportColumns();
      const aData = this._getExportData('visible');
            
      // Create export settings
      const oSettings = {
        workbook: {
          columns: aColumns,
          hierarchyLevel: 'level'
        },
        dataSource: aData,
        fileName: `Projects_Export_${  new Date().toISOString().slice(0, 10)  }.xlsx`,
        worker: true // Use web worker for better performance
      };
            
      // Create spreadsheet and trigger download
      const oSpreadsheet = new Spreadsheet(oSettings);
      oSpreadsheet.build()
        .then(() => {
          MessageToast.show('Excel export completed successfully');
        })
        .catch((oError) => {
          MessageBox.error(`Excel export failed: ${  oError.message}`);
        });
    },
        
    /**
         * Export to PDF with advanced options
         */
    onExportPDFAdvanced: function () {
      if (!this._oPDFExportDialog) {
        Fragment.load({
          id: this.getView().getId(),
          name: 'sap.a2a.view.fragments.PDFExportDialog',
          controller: this
        }).then((oDialog) => {
          this._oPDFExportDialog = oDialog;
          this.getView().addDependent(this._oPDFExportDialog);
                    
          // Initialize PDF export model
          this._initializePDFExport();
                    
          // Open dialog
          this._oPDFExportDialog.open();
        });
      } else {
        // Reset progress
        const oPDFModel = this.getModel('pdfExport');
        oPDFModel.setProperty('/exporting', false);
        oPDFModel.setProperty('/progress', 0);
        this._oPDFExportDialog.open();
      }
    },
        
    /**
         * Initialize PDF export model
         */
    _initializePDFExport: function () {
      const oUserModel = this.getModel('user') || {};
      const oPDFModel = new JSONModel({
        title: `Projects List Report - ${  new Date().toLocaleDateString()}`,
        author: oUserModel.getProperty('/name') || 'A2A Portal User',
        subject: 'A2A Projects Export',
        keywords: 'projects, agents, a2a, export',
        pageSize: 'A4',
        orientation: 'portrait',
        margins: 'normal',
        includePageNumbers: true,
        pageNumberPosition: 'bottom-center',
        includeCoverPage: true,
        includeTOC: true,
        includeHeader: true,
        includeFooter: true,
        includeKPIs: true,
        includeFilters: true,
        includeTable: true,
        includeCharts: false,
        compression: 'fast',
        encrypt: false,
        allowPrint: true,
        allowCopy: true,
        addWatermark: false,
        watermarkText: 'CONFIDENTIAL',
        embedFonts: true,
        subsetFonts: true,
        exporting: false,
        progress: 0,
        progressText: '0%',
        statusMessage: ''
      });
            
      this.setModel(oPDFModel, 'pdfExport');
    },
        
    /**
         * Execute PDF export
         */
    onExecutePDFExport: function () {
      const oPDFModel = this.getModel('pdfExport');
      const oSettings = oPDFModel.getData();
            
      // Validate required fields
      if (!oSettings.title) {
        MessageBox.warning('Please enter a document title.');
        return;
      }
            
      // Start export process
      oPDFModel.setProperty('/exporting', true);
      oPDFModel.setProperty('/progress', 0);
      oPDFModel.setProperty('/statusMessage', 'Initializing PDF export...');
            
      // Simulate PDF generation with jsPDF
      this._generatePDF(oSettings);
    },
        
    /**
         * Generate PDF using jsPDF library
         */
    _generatePDF: function (oSettings) {
      const oPDFModel = this.getModel('pdfExport');
            
      // Update progress
      oPDFModel.setProperty('/progress', 10);
      oPDFModel.setProperty('/progressText', '10%');
      oPDFModel.setProperty('/statusMessage', 'Loading PDF library...');
            
      // Load jsPDF library dynamically
      sap.ui.require(['sap/ui/thirdparty/jsPDF'], (jsPDF) => {
        // Create PDF document
        const doc = new jsPDF({
          orientation: oSettings.orientation,
          unit: 'mm',
          format: oSettings.pageSize.toLowerCase(),
          compress: oSettings.compression !== 'none'
        });
                
        // Set document properties
        doc.setProperties({
          title: oSettings.title,
          author: oSettings.author,
          subject: oSettings.subject,
          keywords: oSettings.keywords,
          creator: 'A2A Agent Portal'
        });
                
        // Update progress
        oPDFModel.setProperty('/progress', 20);
        oPDFModel.setProperty('/progressText', '20%');
        oPDFModel.setProperty('/statusMessage', 'Generating cover page...');
                
        // Add cover page if requested
        if (oSettings.includeCoverPage) {
          this._addCoverPage(doc, oSettings);
        }
                
        // Update progress
        oPDFModel.setProperty('/progress', 30);
        oPDFModel.setProperty('/progressText', '30%');
        oPDFModel.setProperty('/statusMessage', 'Creating table of contents...');
                
        // Add table of contents if requested
        if (oSettings.includeTOC) {
          this._addTableOfContents(doc, oSettings);
        }
                
        // Update progress
        oPDFModel.setProperty('/progress', 40);
        oPDFModel.setProperty('/progressText', '40%');
        oPDFModel.setProperty('/statusMessage', 'Exporting KPI summary...');
                
        // Add KPIs if requested
        if (oSettings.includeKPIs) {
          this._addKPISection(doc, oSettings);
        }
                
        // Update progress
        oPDFModel.setProperty('/progress', 50);
        oPDFModel.setProperty('/progressText', '50%');
        oPDFModel.setProperty('/statusMessage', 'Processing table data...');
                
        // Add table data if requested
        if (oSettings.includeTable) {
          this._addTableData(doc, oSettings);
        }
                
        // Update progress
        oPDFModel.setProperty('/progress', 80);
        oPDFModel.setProperty('/progressText', '80%');
        oPDFModel.setProperty('/statusMessage', 'Applying security settings...');
                
        // Apply encryption if requested
        if (oSettings.encrypt) {
          doc.setEncryption({
            userPermissions: oSettings.allowPrint ? ['print'] : []
          });
        }
                
        // Add watermark if requested
        if (oSettings.addWatermark) {
          this._addWatermark(doc, oSettings.watermarkText);
        }
                
        // Update progress
        oPDFModel.setProperty('/progress', 90);
        oPDFModel.setProperty('/progressText', '90%');
        oPDFModel.setProperty('/statusMessage', 'Finalizing document...');
                
        // Save the PDF
        setTimeout(() => {
          doc.save(`${oSettings.title.replace(/[^a-z0-9]/gi, '_')  }.pdf`);
                    
          // Complete
          oPDFModel.setProperty('/progress', 100);
          oPDFModel.setProperty('/progressText', '100%');
          oPDFModel.setProperty('/statusMessage', 'PDF export completed!');
                    
          setTimeout(() => {
            oPDFModel.setProperty('/exporting', false);
            this._oPDFExportDialog.close();
            MessageToast.show('PDF exported successfully');
          }, 1000);
        }, 500);
                
      }, (_oError) => {
        // Fallback to HTML-based PDF generation
        this._generatePDFFromHTML(oSettings);
      });
    },
        
    /**
         * Add cover page to PDF
         */
    _addCoverPage: function (doc, oSettings) {
      const pageWidth = doc.internal.pageSize.getWidth();
      const _pageHeight = doc.internal.pageSize.getHeight();
            
      // Add logo/header
      doc.setFontSize(24);
      doc.setFont('helvetica', 'bold');
      doc.text('A2A Agent Portal', pageWidth / 2, 50, { align: 'center' });
            
      // Add title
      doc.setFontSize(18);
      doc.setFont('helvetica', 'normal');
      doc.text(oSettings.title, pageWidth / 2, 80, { align: 'center' });
            
      // Add metadata
      doc.setFontSize(12);
      doc.text(`Author: ${  oSettings.author}`, pageWidth / 2, 110, { align: 'center' });
      doc.text(`Date: ${  new Date().toLocaleDateString()}`, pageWidth / 2, 120, { align: 'center' });
            
      // Add new page for content
      doc.addPage();
    },
        
    /**
         * Add table of contents
         */
    _addTableOfContents: function (doc, oSettings) {
      doc.setFontSize(16);
      doc.setFont('helvetica', 'bold');
      doc.text('Table of Contents', 20, 30);
            
      doc.setFontSize(12);
      doc.setFont('helvetica', 'normal');
            
      let yPos = 50;
      let pageNum = 3;
            
      if (oSettings.includeKPIs) {
        doc.text('1. Key Performance Indicators', 25, yPos);
        doc.text(pageNum.toString(), 180, yPos, { align: 'right' });
        yPos += 10;
        pageNum++;
      }
            
      if (oSettings.includeFilters) {
        doc.text('2. Active Filters', 25, yPos);
        doc.text(pageNum.toString(), 180, yPos, { align: 'right' });
        yPos += 10;
        pageNum++;
      }
            
      if (oSettings.includeTable) {
        doc.text('3. Project Data', 25, yPos);
        doc.text(pageNum.toString(), 180, yPos, { align: 'right' });
        yPos += 10;
      }
            
      doc.addPage();
    },
        
    /**
         * Add KPI section
         */
    _addKPISection: function (doc, _oSettings) {
      const oViewModel = this.getModel('listReport');
      const oKPIs = oViewModel.getProperty('/kpi');
            
      doc.setFontSize(16);
      doc.setFont('helvetica', 'bold');
      doc.text('Key Performance Indicators', 20, 30);
            
      doc.setFontSize(12);
      doc.setFont('helvetica', 'normal');
            
      // Draw KPI boxes
      let xPos = 20;
      const yPos = 50;
      const boxWidth = 80;
      const boxHeight = 30;
            
      // Total Projects
      doc.rect(xPos, yPos, boxWidth, boxHeight);
      doc.text('Total Projects', xPos + 5, yPos + 10);
      doc.setFont('helvetica', 'bold');
      doc.text(oKPIs.totalProjects.toString(), xPos + boxWidth / 2, yPos + 20, { align: 'center' });
      doc.setFont('helvetica', 'normal');
            
      // Active Projects
      xPos = 110;
      doc.rect(xPos, yPos, boxWidth, boxHeight);
      doc.text('Active Projects', xPos + 5, yPos + 10);
      doc.setFont('helvetica', 'bold');
      doc.text(oKPIs.activeProjects.toString(), xPos + boxWidth / 2, yPos + 20, { align: 'center' });
      doc.setFont('helvetica', 'normal');
            
      doc.addPage();
    },
        
    /**
         * Add table data to PDF
         */
    _addTableData: function (doc, _oSettings) {
      const aData = this._getExportData('visible');
      const aColumns = this._getExportColumns();
            
      doc.setFontSize(16);
      doc.setFont('helvetica', 'bold');
      doc.text('Project Data', 20, 30);
            
      // Use autoTable plugin if available
      if (doc.autoTable) {
        const headers = aColumns.map(col => col.label);
        const rows = aData.map(item => {
          return aColumns.map(col => {
            const value = item[col.property];
            if (value instanceof Date) {
              return value.toLocaleDateString();
            }
            return value || '';
          });
        });
                
        doc.autoTable({
          head: [headers],
          body: rows,
          startY: 40,
          theme: 'grid',
          styles: {
            fontSize: 9,
            cellPadding: 2
          },
          headStyles: {
            fillColor: [66, 139, 202],
            textColor: 255
          }
        });
      } else {
        // Manual table rendering
        this._renderTableManually(doc, aColumns, aData, 40);
      }
    },
        
    /**
         * Add watermark to all pages
         */
    _addWatermark: function (doc, text) {
      const totalPages = doc.internal.getNumberOfPages();
            
      for (let i = 1; i <= totalPages; i++) {
        doc.setPage(i);
        doc.setFontSize(50);
        doc.setTextColor(200);
        doc.setFont('helvetica', 'bold');
                
        // Diagonal watermark
        doc.text(text, doc.internal.pageSize.getWidth() / 2, 
          doc.internal.pageSize.getHeight() / 2, 
          { angle: 45, align: 'center' });
                
        doc.setTextColor(0); // Reset to black
      }
    },
        
    /**
         * Fallback HTML-based PDF generation
         */
    _generatePDFFromHTML: function (oSettings) {
      const oPDFModel = this.getModel('pdfExport');
            
      // Create print window with PDF-specific styles
      const oPrintWindow = window.open('', '_blank');
      const sHTML = this._generatePDFHTML(oSettings);
            
      oPrintWindow.document.write(sHTML);
      oPrintWindow.document.close();
            
      // Update progress
      oPDFModel.setProperty('/progress', 100);
      oPDFModel.setProperty('/progressText', '100%');
      oPDFModel.setProperty('/statusMessage', 'Opening print dialog for PDF...');
            
      setTimeout(() => {
        oPrintWindow.print();
        oPDFModel.setProperty('/exporting', false);
        this._oPDFExportDialog.close();
                
        MessageBox.information(
          "Please select 'Save as PDF' in the print dialog to save the document.",
          {
            title: 'PDF Export',
            actions: [MessageBox.Action.OK]
          }
        );
      }, 1000);
    },
        
    /**
         * Generate HTML for PDF export
         */
    _generatePDFHTML: function (oSettings) {
      const aData = this._getExportData('visible');
      const aColumns = this._getExportColumns();
            
      let sHTML = `<!DOCTYPE html>
<html>
<head>
    <title>${oSettings.title}</title>
    <meta charset="UTF-8">
    <meta name="author" content="${oSettings.author}">
    <meta name="subject" content="${oSettings.subject}">
    <meta name="keywords" content="${oSettings.keywords}">
    <style>
        @page {
            size: ${oSettings.pageSize} ${oSettings.orientation};
            margin: ${oSettings.margins === 'normal' ? '2.5cm' : 
    oSettings.margins === 'narrow' ? '1.25cm' : '3.8cm'};
            @bottom-center {
                content: ${oSettings.includePageNumbers ? 'counter(page)' : ''};
            }
        }
        body {
            font-family: Arial, sans-serif;
            font-size: 10pt;
            line-height: 1.5;
        }
        .cover-page {
            page-break-after: always;
            text-align: center;
            padding-top: 30%;
        }
        .cover-page h1 {
            font-size: 24pt;
            margin-bottom: 20px;
        }
        .cover-page h2 {
            font-size: 18pt;
            font-weight: normal;
            margin-bottom: 50px;
        }
        .toc {
            page-break-after: always;
        }
        .kpi-section {
            page-break-after: always;
        }
        .kpi-box {
            display: inline-block;
            border: 1px solid #ddd;
            padding: 20px;
            margin: 10px;
            text-align: center;
            width: 150px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #428bca;
            color: white;
            font-weight: bold;
        }
        tr:nth-child(even) {
            background-color: #f5f5f5;
        }
        .watermark {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%) rotate(-45deg);
            font-size: 100px;
            color: rgba(0,0,0,0.1);
            z-index: -1;
        }
        @media print {
            .no-print {
                display: none;
            }
        }
    </style>
</head>
<body>`;
            
      // Add watermark if requested
      if (oSettings.addWatermark) {
        sHTML += `<div class="watermark">${oSettings.watermarkText}</div>`;
      }
            
      // Add cover page if requested
      if (oSettings.includeCoverPage) {
        sHTML += `
    <div class="cover-page">
        <h1>A2A Agent Portal</h1>
        <h2>${oSettings.title}</h2>
        <p>Author: ${oSettings.author}</p>
        <p>Date: ${new Date().toLocaleDateString()}</p>
    </div>`;
      }
            
      // Add table of contents if requested
      if (oSettings.includeTOC) {
        sHTML += `
    <div class="toc">
        <h2>Table of Contents</h2>
        <ol>`;
        if (oSettings.includeKPIs) {
          sHTML += `<li>Key Performance Indicators</li>`;
        }
        if (oSettings.includeFilters) {
          sHTML += `<li>Active Filters</li>`;
        }
        if (oSettings.includeTable) {
          sHTML += `<li>Project Data</li>`;
        }
        sHTML += `
        </ol>
    </div>`;
      }
            
      // Add KPIs if requested
      if (oSettings.includeKPIs) {
        const oViewModel = this.getModel('listReport');
        const oKPIs = oViewModel.getProperty('/kpi');
                
        sHTML += `
    <div class="kpi-section">
        <h2>Key Performance Indicators</h2>
        <div class="kpi-box">
            <h3>Total Projects</h3>
            <p style="font-size: 24pt; font-weight: bold;">${oKPIs.totalProjects}</p>
        </div>
        <div class="kpi-box">
            <h3>Active Projects</h3>
            <p style="font-size: 24pt; font-weight: bold;">${oKPIs.activeProjects}</p>
        </div>
        <div class="kpi-box">
            <h3>Completed Projects</h3>
            <p style="font-size: 24pt; font-weight: bold;">${oKPIs.completedProjects}</p>
        </div>
        <div class="kpi-box">
            <h3>Success Rate</h3>
            <p style="font-size: 24pt; font-weight: bold;">${oKPIs.deploymentSuccessRate}%</p>
        </div>
    </div>`;
      }
            
      // Add table data if requested
      if (oSettings.includeTable) {
        sHTML += `
    <div class="data-section">
        <h2>Project Data</h2>
        <table>
            <thead>
                <tr>`;
                
        aColumns.forEach(col => {
          sHTML += `<th>${col.label}</th>`;
        });
                
        sHTML += `
                </tr>
            </thead>
            <tbody>`;
                
        aData.forEach(row => {
          sHTML += '<tr>';
          aColumns.forEach(col => {
            let value = row[col.property] || '';
            if (value instanceof Date) {
              value = value.toLocaleDateString();
            }
            sHTML += `<td>${value}</td>`;
          });
          sHTML += '</tr>';
        });
                
        sHTML += `
            </tbody>
        </table>
    </div>`;
      }
            
      sHTML += `
</body>
</html>`;
            
      return sHTML;
    },
        
    /**
         * Preview PDF before export
         */
    onPDFPreview: function () {
      const oPDFModel = this.getModel('pdfExport');
      const oSettings = oPDFModel.getData();
            
      // Create preview window
      const oPreviewWindow = window.open('', '_blank');
      const sHTML = this._generatePDFHTML(oSettings);
            
      oPreviewWindow.document.write(sHTML);
      oPreviewWindow.document.close();
    },
        
    /**
         * Close PDF export dialog
         */
    onClosePDFExport: function () {
      if (this._oPDFExportDialog) {
        this._oPDFExportDialog.close();
      }
    },
        
    /**
         * Export to CSV - opens advanced options dialog
         */
    onExportCSV: function () {
      // Initialize CSV export model if not exists
      if (!this._oCSVExportModel) {
        this._oCSVExportModel = new JSONModel({
          fileName: `projects_export_${  new Date().toISOString().slice(0, 10)}`,
          fileNamePreview: `projects_export_${  new Date().toISOString().slice(0, 10)}`,
          delimiterIndex: 0, // Comma by default
          qualifierIndex: 0, // Double quotes by default
          lineEnding: 'CRLF',
          encoding: 'UTF-8',
          includeHeaders: true,
          includeHidden: false,
          useFormattedValues: true,
          includeRowNumbers: false,
          includeSummary: false,
          includeMetadata: false,
          dataRange: 0, // Visible data by default
          rowLimit: '',
          escapeSpecial: true,
          nullToEmpty: true,
          trimWhitespace: false,
          useLocale: true,
          dateFormat: 'ISO',
          customDateFormat: ''
        });
        this.setModel(this._oCSVExportModel, 'csvExport');
      }
            
      // Open CSV export dialog
      if (!this._oCSVExportDialog) {
        Fragment.load({
          id: this.getView().getId(),
          name: 'sap.a2a.view.fragments.CSVExportDialog',
          controller: this
        }).then((oDialog) => {
          this._oCSVExportDialog = oDialog;
          this.getView().addDependent(this._oCSVExportDialog);
          this._oCSVExportDialog.open();
        });
      } else {
        this._oCSVExportDialog.open();
      }
    },
        
    /**
         * Handle CSV file name change
         */
    onCSVFileNameChange: function (oEvent) {
      const sValue = oEvent.getParameter('value');
      const sCleanName = sValue.replace(/[^a-zA-Z0-9_-]/g, '_');
      this._oCSVExportModel.setProperty('/fileNamePreview', sCleanName);
    },
        
    /**
         * Execute CSV export with selected options
         */
    onExecuteCSVExport: function () {
      const oModel = this._oCSVExportModel;
      const oData = oModel.getData();
            
      // Determine data scope
      let sScope = 'visible';
      switch (oData.dataRange) {
      case 1: sScope = 'selected'; break;
      case 2: sScope = 'filtered'; break;
      case 3: sScope = 'all'; break;
      }
            
      // Get data to export
      const aData = this._getExportData(sScope);
      const aColumns = this._getExportColumns();
            
      // Apply row limit if specified
      let aExportData = aData;
      if (oData.rowLimit && parseInt(oData.rowLimit) > 0) {
        aExportData = aData.slice(0, parseInt(oData.rowLimit));
      }
            
      // Generate enhanced CSV content
      const sCSV = this._generateEnhancedCSV(aColumns, aExportData, oData);
            
      // Handle encoding
      let blob;
      if (oData.encoding === 'UTF-8-BOM') {
        // Add BOM for Excel compatibility
        const BOM = '\uFEFF';
        blob = new Blob([BOM + sCSV], { type: 'text/csv;charset=utf-8;' });
      } else {
        blob = new Blob([sCSV], { type: `text/csv;charset=${  oData.encoding  };` });
      }
            
      // Download file
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.setAttribute('href', url);
      link.setAttribute('download', `${oData.fileNamePreview  }.csv`);
      link.style.visibility = 'hidden';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
            
      MessageToast.show('CSV export completed successfully');
      this._oCSVExportDialog.close();
    },
        
    /**
         * Generate enhanced CSV with advanced options
         */
    _generateEnhancedCSV: function (aColumns, aData, oOptions) {
      let sCSV = '';
            
      // Delimiters
      const delimiters = [',', ';', '\t', '|'];
      const delimiter = delimiters[oOptions.delimiterIndex];
            
      // Text qualifiers
      const qualifiers = ['"', "'", ''];
      const qualifier = qualifiers[oOptions.qualifierIndex];
            
      // Line endings
      const lineEndings = { 'CRLF': '\r\n', 'LF': '\n', 'CR': '\r' };
      const lineEnding = lineEndings[oOptions.lineEnding];
            
      // Add metadata if requested
      if (oOptions.includeMetadata) {
        sCSV += `Export Date${  delimiter  }${new Date().toLocaleString()  }${lineEnding}`;
        sCSV += `Total Records${  delimiter  }${aData.length  }${lineEnding}`;
        sCSV += `Export Type${  delimiter  }${['Visible', 'Selected', 'Filtered', 'All'][oOptions.dataRange]  }${lineEnding}`;
        sCSV += lineEnding; // Empty line
      }
            
      // Filter columns based on options
      let exportColumns = aColumns;
      if (!oOptions.includeHidden) {
        exportColumns = aColumns.filter(col => !col.hidden);
      }
            
      // Add headers if requested
      if (oOptions.includeHeaders) {
        if (oOptions.includeRowNumbers) {
          sCSV += this._escapeCSVValue('#', qualifier, delimiter);
          sCSV += delimiter;
        }
                
        sCSV += exportColumns.map(col => {
          return this._escapeCSVValue(col.label, qualifier, delimiter);
        }).join(delimiter) + lineEnding;
      }
            
      // Add data rows
      aData.forEach((oRow, index) => {
        const aRowData = [];
                
        if (oOptions.includeRowNumbers) {
          aRowData.push(this._escapeCSVValue(index + 1, qualifier, delimiter));
        }
                
        exportColumns.forEach(oCol => {
          let value = oRow[oCol.property];
                    
          // Handle null values
          if (value === null || value === undefined) {
            value = oOptions.nullToEmpty ? '' : 'NULL';
          }
                    
          // Format dates
          if (value instanceof Date) {
            value = this._formatDateForCSV(value, oOptions.dateFormat, oOptions.customDateFormat);
          }
                    
          // Use formatted values if requested
          if (oOptions.useFormattedValues && oCol.formatter) {
            value = oCol.formatter(value);
          }
                    
          // Convert to string
          value = String(value);
                    
          // Trim whitespace if requested
          if (oOptions.trimWhitespace) {
            value = value.trim();
          }
                    
          // Escape value
          value = this._escapeCSVValue(value, qualifier, delimiter, oOptions.escapeSpecial);
                    
          aRowData.push(value);
        });
                
        sCSV += aRowData.join(delimiter) + lineEnding;
      });
            
      // Add summary row if requested
      if (oOptions.includeSummary) {
        const summaryRow = [];
                
        if (oOptions.includeRowNumbers) {
          summaryRow.push(this._escapeCSVValue('Total', qualifier, delimiter));
        }
                
        exportColumns.forEach(oCol => {
          if (oCol.property === 'name') {
            summaryRow.push(this._escapeCSVValue(`Total: ${  aData.length  } records`, qualifier, delimiter));
          } else if (oCol.type === 'number') {
            // Calculate sum for numeric columns
            const sum = aData.reduce((acc, row) => acc + (parseFloat(row[oCol.property]) || 0), 0);
            summaryRow.push(this._escapeCSVValue(sum, qualifier, delimiter));
          } else {
            summaryRow.push(this._escapeCSVValue('', qualifier, delimiter));
          }
        });
                
        sCSV += summaryRow.join(delimiter) + lineEnding;
      }
            
      return sCSV;
    },
        
    /**
         * Escape CSV value
         */
    _escapeCSVValue: function (value, qualifier, delimiter, escapeSpecial) {
      if (qualifier === '') {
        // No qualifier - just escape delimiter
        return String(value).replace(new RegExp(delimiter, 'g'), `\\${  delimiter}`);
      }
            
      value = String(value);
            
      // Check if value needs escaping
      const needsEscape = value.includes(delimiter) || 
                               value.includes(qualifier) || 
                               value.includes('\n') || 
                               value.includes('\r');
            
      if (needsEscape || escapeSpecial) {
        // Escape qualifier by doubling it
        value = value.replace(new RegExp(qualifier, 'g'), qualifier + qualifier);
        // Wrap in qualifiers
        return qualifier + value + qualifier;
      }
            
      return value;
    },
        
    /**
         * Format date for CSV export
         */
    _formatDateForCSV: function (date, format, customFormat) {
      switch (format) {
      case 'ISO':
        return date.toISOString().slice(0, 10);
      case 'US':
        return `${(date.getMonth() + 1).toString().padStart(2, '0')  }/${ 
          date.getDate().toString().padStart(2, '0')  }/${ 
          date.getFullYear()}`;
      case 'EU':
        return `${date.getDate().toString().padStart(2, '0')  }/${ 
          (date.getMonth() + 1).toString().padStart(2, '0')  }/${ 
          date.getFullYear()}`;
      case 'CUSTOM':
        return this._formatCustomDate(date, customFormat);
      default:
        return date.toLocaleDateString();
      }
    },
        
    /**
         * Format custom date
         */
    _formatCustomDate: function (date, format) {
      const replacements = {
        'YYYY': date.getFullYear(),
        'YY': date.getFullYear().toString().slice(-2),
        'MM': (date.getMonth() + 1).toString().padStart(2, '0'),
        'M': date.getMonth() + 1,
        'DD': date.getDate().toString().padStart(2, '0'),
        'D': date.getDate()
      };
            
      let result = format;
      Object.keys(replacements).forEach(key => {
        result = result.replace(new RegExp(key, 'g'), replacements[key]);
      });
            
      return result;
    },
        
    /**
         * Preview CSV export
         */
    onCSVPreview: function () {
      const oModel = this._oCSVExportModel;
      const oData = oModel.getData();
            
      // Get sample data (first 10 rows)
      let sScope = 'visible';
      switch (oData.dataRange) {
      case 1: sScope = 'selected'; break;
      case 2: sScope = 'filtered'; break;
      case 3: sScope = 'all'; break;
      }
            
      const aData = this._getExportData(sScope).slice(0, 10);
      const aColumns = this._getExportColumns();
            
      // Generate preview
      const sPreview = this._generateEnhancedCSV(aColumns, aData, oData);
            
      // Show preview in dialog
      const oPreviewDialog = new sap.m.Dialog({
        title: 'CSV Export Preview (First 10 rows)',
        contentWidth: '800px',
        contentHeight: '400px',
        resizable: true,
        content: [
          new sap.m.TextArea({
            value: sPreview,
            width: '100%',
            height: '100%',
            editable: false,
            growing: false
          })
        ],
        buttons: [
          new sap.m.Button({
            text: 'Close',
            press: function () {
              oPreviewDialog.close();
            }
          })
        ],
        afterClose: function () {
          oPreviewDialog.destroy();
        }
      });
            
      oPreviewDialog.open();
    },
        
    /**
         * Close CSV export dialog
         */
    onCloseCSVExport: function () {
      if (this._oCSVExportDialog) {
        this._oCSVExportDialog.close();
      }
    },
        
    /**
         * Handle large export operations
         */
    onLargeExport: function (sFormat) {
      const aData = this._getExportData('all');
      const nTotalRecords = aData.length;
            
      // Check if this is a large export
      if (nTotalRecords > 1000) {
        MessageBox.confirm(
          `You are about to export ${nTotalRecords} records. Large exports may take several minutes and consume significant memory. Do you want to continue?`,
          {
            title: 'Large Export Warning',
            actions: [MessageBox.Action.YES, MessageBox.Action.NO],
            emphasizedAction: MessageBox.Action.NO,
            onClose: function (oAction) {
              if (oAction === MessageBox.Action.YES) {
                this._performLargeExport(sFormat, aData);
              }
            }.bind(this)
          }
        );
      } else {
        // Regular export for smaller datasets
        this._proceedWithExport(sFormat);
      }
    },
        
    /**
         * Perform large export with progress tracking
         */
    _performLargeExport: function (sFormat, aData) {
      // Initialize large export model
      if (!this._oLargeExportModel) {
        this._oLargeExportModel = new JSONModel({
          inProgress: true,
          currentChunk: 0,
          totalChunks: 0,
          processedRecords: 0,
          totalRecords: aData.length,
          format: sFormat,
          startTime: new Date(),
          estimatedTime: 0,
          chunks: []
        });
        this.setModel(this._oLargeExportModel, 'largeExport');
      }
            
      // Open large export progress dialog
      if (!this._oLargeExportDialog) {
        Fragment.load({
          id: this.getView().getId(),
          name: 'sap.a2a.view.fragments.LargeExportDialog',
          controller: this
        }).then((oDialog) => {
          this._oLargeExportDialog = oDialog;
          this.getView().addDependent(this._oLargeExportDialog);
          this._oLargeExportDialog.open();
          this._processLargeExportChunks(sFormat, aData);
        });
      } else {
        this._oLargeExportDialog.open();
        this._processLargeExportChunks(sFormat, aData);
      }
    },
        
    /**
         * Process large export in chunks
         */
    _processLargeExportChunks: function (sFormat, aData) {
      const oModel = this._oLargeExportModel;
      const chunkSize = 500; // Process 500 records at a time
      const totalChunks = Math.ceil(aData.length / chunkSize);
            
      oModel.setProperty('/totalChunks', totalChunks);
      oModel.setProperty('/inProgress', true);
            
      const aColumns = this._getExportColumns();
      const _chunks = [];
            
      // Process chunks with web worker if available
      if (window.Worker && sFormat === 'csv') {
        this._processWithWebWorker(sFormat, aData, aColumns, chunkSize);
      } else {
        // Fallback to sequential processing
        this._processSequentially(sFormat, aData, aColumns, chunkSize);
      }
    },
        
    /**
         * Process export with web worker for better performance
         */
    _processWithWebWorker: function (sFormat, aData, aColumns, chunkSize) {
      // Create inline web worker
      const workerCode = `
                self.onmessage = function(e) {
                    const { chunk, columns, chunkIndex } = e.data;
                    let result = '';
                    
                    // Process chunk
                    chunk.forEach(row => {
                        const rowData = [];
                        columns.forEach(col => {
                            let value = row[col.property] || '';
                            if (typeof value === 'string') {
                                value = '"' + value.replace(/"/g, '""') + '"';
                            }
                            rowData.push(value);
                        });
                        result += rowData.join(',') + '\\n';
                    });
                    
                    self.postMessage({ 
                        chunkIndex: chunkIndex, 
                        result: result,
                        recordsProcessed: chunk.length
                    });
                };
            `;
            
      const blob = new Blob([workerCode], { type: 'application/javascript' });
      const worker = new Worker(URL.createObjectURL(blob));
            
      const oModel = this._oLargeExportModel;
      const chunks = [];
      let processedChunks = 0;
            
      worker.onmessage = function(e) {
        chunks[e.data.chunkIndex] = e.data.result;
        processedChunks++;
                
        const processedRecords = oModel.getProperty('/processedRecords') + e.data.recordsProcessed;
        oModel.setProperty('/processedRecords', processedRecords);
        oModel.setProperty('/currentChunk', processedChunks);
                
        // Update time estimate
        const elapsed = new Date() - oModel.getProperty('/startTime');
        const rate = processedRecords / elapsed;
        const remaining = (aData.length - processedRecords) / rate;
        oModel.setProperty('/estimatedTime', Math.ceil(remaining / 1000));
                
        if (processedChunks === oModel.getProperty('/totalChunks')) {
          // All chunks processed
          worker.terminate();
          this._finalizeLargeExport(sFormat, chunks, aColumns);
        }
      }.bind(this);
            
      // Send chunks to worker
      for (let i = 0; i < aData.length; i += chunkSize) {
        const chunk = aData.slice(i, i + chunkSize);
        worker.postMessage({
          chunk: chunk,
          columns: aColumns,
          chunkIndex: Math.floor(i / chunkSize)
        });
      }
    },
        
    /**
         * Fallback sequential processing
         */
    _processSequentially: function (sFormat, aData, aColumns, chunkSize) {
      const oModel = this._oLargeExportModel;
      const chunks = [];
      let currentChunk = 0;
            
      const processNextChunk = () => {
        const startIdx = currentChunk * chunkSize;
        const endIdx = Math.min(startIdx + chunkSize, aData.length);
        const chunk = aData.slice(startIdx, endIdx);
                
        // Process chunk based on format
        let chunkResult;
        switch (sFormat) {
        case 'csv':
          chunkResult = this._generateCSVChunk(chunk, aColumns, currentChunk === 0);
          break;
        case 'excel':
          chunkResult = chunk; // Excel handles its own chunking
          break;
        case 'pdf':
          chunkResult = this._generatePDFChunk(chunk, aColumns);
          break;
        }
                
        chunks.push(chunkResult);
        oModel.setProperty('/processedRecords', endIdx);
        oModel.setProperty('/currentChunk', currentChunk + 1);
                
        // Update time estimate
        const elapsed = new Date() - oModel.getProperty('/startTime');
        const rate = endIdx / elapsed;
        const remaining = (aData.length - endIdx) / rate;
        oModel.setProperty('/estimatedTime', Math.ceil(remaining / 1000));
                
        currentChunk++;
                
        if (currentChunk < oModel.getProperty('/totalChunks')) {
          // Process next chunk with small delay to keep UI responsive
          setTimeout(processNextChunk, 10);
        } else {
          // All chunks processed
          this._finalizeLargeExport(sFormat, chunks, aColumns);
        }
      };
            
      // Start processing
      processNextChunk();
    },
        
    /**
         * Generate CSV chunk
         */
    _generateCSVChunk: function (aData, aColumns, includeHeaders) {
      let sCSV = '';
            
      // Add headers for first chunk
      if (includeHeaders) {
        sCSV += `${aColumns.map(col => `"${  col.label  }"`).join(',')  }\n`;
      }
            
      // Add data rows
      aData.forEach(oRow => {
        const aRowData = [];
        aColumns.forEach(oCol => {
          let value = oRow[oCol.property] || '';
                    
          if (value instanceof Date) {
            value = value.toLocaleDateString();
          }
                    
          if (typeof value === 'string') {
            value = `"${  value.replace(/"/g, '""')  }"`;
          }
                    
          aRowData.push(value);
        });
        sCSV += `${aRowData.join(',')  }\n`;
      });
            
      return sCSV;
    },
        
    /**
         * Generate PDF chunk (simplified)
         */
    _generatePDFChunk: function (aData, aColumns) {
      // Return data formatted for PDF generation
      return {
        columns: aColumns.map(col => col.label),
        rows: aData.map(row => {
          return aColumns.map(col => {
            const value = row[col.property];
            if (value instanceof Date) {
              return value.toLocaleDateString();
            }
            return String(value || '');
          });
        })
      };
    },
        
    /**
         * Finalize large export
         */
    _finalizeLargeExport: function (sFormat, chunks, aColumns) {
      const oModel = this._oLargeExportModel;
      oModel.setProperty('/inProgress', false);
            
      try {
        switch (sFormat) {
        case 'csv': {
          // Combine CSV chunks
          const csvContent = chunks.join('');
          const csvBlob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
          this._downloadBlob(csvBlob, `large_export_${  new Date().toISOString().slice(0, 10)  }.csv`);
          break;
        }
                        
        case 'excel': {
          // Use SAP export library for Excel
          const allData = chunks.flat();
          const oSettings = {
            workbook: {
              columns: aColumns,
              context: {
                title: 'Large Dataset Export',
                modifiedBy: this.getModel('user').getProperty('/name'),
                sheetName: 'Export Data'
              }
            },
            dataSource: allData,
            fileName: `large_export_${  new Date().toISOString().slice(0, 10)  }.xlsx`,
            worker: false // Already processed in chunks
          };
                        
          const oSpreadsheet = new Spreadsheet(oSettings);
          oSpreadsheet.build()
            .then(() => {
              MessageToast.show('Large Excel export completed successfully');
              this._oLargeExportDialog.close();
            })
            .catch((oError) => {
              MessageBox.error(`Excel export failed: ${  oError.message}`);
            });
          return;
        }
                        
        case 'pdf':
          // Handle PDF generation for large datasets
          this._generateLargePDF(chunks, aColumns);
          break;
        }
                
        const elapsed = new Date() - oModel.getProperty('/startTime');
        MessageToast.show(`Export completed in ${Math.round(elapsed / 1000)} seconds`);
                
        if (this._oLargeExportDialog) {
          this._oLargeExportDialog.close();
        }
                
      } catch (error) {
        MessageBox.error(`Export failed: ${  error.message}`);
        oModel.setProperty('/inProgress', false);
      }
    },
        
    /**
         * Download blob with proper cleanup
         */
    _downloadBlob: function (blob, filename) {
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.setAttribute('href', url);
      link.setAttribute('download', filename);
      link.style.visibility = 'hidden';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
            
      // Clean up object URL after download
      setTimeout(() => URL.revokeObjectURL(url), 100);
    },
        
    /**
         * Cancel large export
         */
    onCancelLargeExport: function () {
      if (this._oLargeExportModel) {
        this._oLargeExportModel.setProperty('/inProgress', false);
      }
      if (this._oLargeExportDialog) {
        this._oLargeExportDialog.close();
      }
      MessageToast.show('Export cancelled');
    },
        
    /**
         * Export only selected items
         */
    onExportSelected: function () {
      const oTable = this.byId('projectsTable');
      const aSelectedItems = oTable.getSelectedItems();
            
      if (aSelectedItems.length === 0) {
        MessageBox.warning('Please select at least one item to export.');
        return;
      }
            
      const aColumns = this._getExportColumns();
      const aData = this._getExportData('selected');
            
      const oSettings = {
        workbook: {
          columns: aColumns,
          context: {
            title: 'Selected Projects Export',
            modifiedBy: this.getModel('user').getProperty('/name'),
            sheetName: 'Selected Projects'
          }
        },
        dataSource: aData,
        fileName: `Selected_Projects_${  new Date().toISOString().slice(0, 10)  }.xlsx`,
        worker: true
      };
            
      const oSpreadsheet = new Spreadsheet(oSettings);
      oSpreadsheet.build()
        .then(() => {
          MessageToast.show(`${aSelectedItems.length  } selected items exported successfully`);
        })
        .catch((oError) => {
          MessageBox.error(`Export failed: ${  oError.message}`);
        });
    },
        
    /**
         * Export filtered data
         */
    onExportFiltered: function () {
      const oBinding = this.byId('projectsTable').getBinding('items');
      const aFilters = oBinding.aFilters;
            
      if (!aFilters || aFilters.length === 0) {
        MessageBox.information('No filters applied. Exporting all visible data.');
      }
            
      const aColumns = this._getExportColumns();
      const aData = this._getExportData('filtered');
            
      const oSettings = {
        workbook: {
          columns: aColumns,
          context: {
            title: 'Filtered Projects Export',
            modifiedBy: this.getModel('user').getProperty('/name'),
            sheetName: 'Filtered Projects',
            metaSheetName: 'Export Information'
          }
        },
        dataSource: aData,
        fileName: `Filtered_Projects_${  new Date().toISOString().slice(0, 10)  }.xlsx`,
        worker: true
      };
            
      const oSpreadsheet = new Spreadsheet(oSettings);
      oSpreadsheet.build()
        .then(() => {
          MessageToast.show('Filtered data exported successfully');
        })
        .catch((oError) => {
          MessageBox.error(`Export failed: ${  oError.message}`);
        });
    },
        
    /**
         * Export all data (bypass filters)
         */
    onExportAll: function () {
      MessageBox.confirm(
        'This will export all projects regardless of current filters. Continue?',
        {
          title: 'Export All Data',
          actions: [MessageBox.Action.YES, MessageBox.Action.NO],
          emphasizedAction: MessageBox.Action.YES,
          onClose: function(oAction) {
            if (oAction === MessageBox.Action.YES) {
              this._performFullExport();
            }
          }.bind(this)
        }
      );
    },
        
    /**
         * Perform full export of all data
         */
    _performFullExport: function () {
      const aColumns = this._getExportColumns(true); // Include all columns
      const aData = this._getExportData('all');
            
      // Show progress dialog for large exports
      const oBusyDialog = new sap.m.BusyDialog({
        text: 'Preparing export...'
      });
      oBusyDialog.open();
            
      const oSettings = {
        workbook: {
          columns: aColumns,
          context: {
            title: 'Complete Projects Export',
            modifiedBy: this.getModel('user').getProperty('/name'),
            sheetName: 'All Projects',
            application: 'A2A Agent Portal',
            metaSheetName: 'Export Metadata'
          }
        },
        dataSource: aData,
        fileName: `All_Projects_${  new Date().toISOString().slice(0, 10)  }.xlsx`,
        worker: true
      };
            
      const oSpreadsheet = new Spreadsheet(oSettings);
      oSpreadsheet.build()
        .then(() => {
          oBusyDialog.close();
          MessageToast.show('Complete export finished successfully');
        })
        .catch((oError) => {
          oBusyDialog.close();
          MessageBox.error(`Export failed: ${  oError.message}`);
        });
    },
        
    /**
         * Get export columns configuration
         */
    _getExportColumns: function (bIncludeAll) {
      const aColumns = [];
            
      // Basic columns
      aColumns.push({
        label: 'Project Name',
        property: 'name',
        type: exportLibrary.EdmType.String,
        width: 30
      });
            
      aColumns.push({
        label: 'Description',
        property: 'description',
        type: exportLibrary.EdmType.String,
        width: 50
      });
            
      aColumns.push({
        label: 'Status',
        property: 'status',
        type: exportLibrary.EdmType.String,
        width: 15
      });
            
      aColumns.push({
        label: 'Priority',
        property: 'priority',
        type: exportLibrary.EdmType.String,
        width: 15
      });
            
      aColumns.push({
        label: 'Department',
        property: 'department',
        type: exportLibrary.EdmType.String,
        width: 20
      });
            
      aColumns.push({
        label: 'Project Manager',
        property: 'projectManager',
        type: exportLibrary.EdmType.String,
        width: 25
      });
            
      aColumns.push({
        label: 'Start Date',
        property: 'startDate',
        type: exportLibrary.EdmType.Date,
        width: 15,
        format: 'dd.mm.yyyy'
      });
            
      aColumns.push({
        label: 'End Date',
        property: 'endDate',
        type: exportLibrary.EdmType.Date,
        width: 15,
        format: 'dd.mm.yyyy'
      });
            
      aColumns.push({
        label: 'Budget',
        property: 'budget',
        type: exportLibrary.EdmType.Number,
        width: 15,
        scale: 2,
        delimiter: true
      });
            
      aColumns.push({
        label: 'Currency',
        property: 'currency',
        type: exportLibrary.EdmType.String,
        width: 10
      });
            
      // Additional columns if requested
      if (bIncludeAll) {
        aColumns.push({
          label: 'Agents Count',
          property: 'agentCount',
          type: exportLibrary.EdmType.Number,
          width: 15
        });
                
        aColumns.push({
          label: 'Last Deployment',
          property: 'lastDeployment',
          type: exportLibrary.EdmType.DateTime,
          width: 20,
          format: 'dd.mm.yyyy hh:mm'
        });
                
        aColumns.push({
          label: 'Created By',
          property: 'createdBy',
          type: exportLibrary.EdmType.String,
          width: 20
        });
                
        aColumns.push({
          label: 'Created Date',
          property: 'createdDate',
          type: exportLibrary.EdmType.DateTime,
          width: 20,
          format: 'dd.mm.yyyy hh:mm'
        });
                
        aColumns.push({
          label: 'Modified By',
          property: 'modifiedBy',
          type: exportLibrary.EdmType.String,
          width: 20
        });
                
        aColumns.push({
          label: 'Modified Date',
          property: 'modifiedDate',
          type: exportLibrary.EdmType.DateTime,
          width: 20,
          format: 'dd.mm.yyyy hh:mm'
        });
      }
            
      return aColumns;
    },
        
    /**
         * Get export data based on scope
         */
    _getExportData: function (sScope) {
      const oTable = this.byId('projectsTable');
      const aData = [];
      let aItems = [];
            
      switch (sScope) {
      case 'selected':
        aItems = oTable.getSelectedItems();
        break;
      case 'filtered':
      case 'visible':
        aItems = oTable.getItems();
        break;
      case 'all': {
        // Get all data from model
        const oModel = this.getModel();
        const aAllProjects = oModel.getProperty('/Projects') || [];
        aAllProjects.forEach(oProject => {
          aData.push(this._prepareExportItem(oProject));
        });
        return aData;
      }
      }
            
      // Process items
      aItems.forEach(oItem => {
        const oContext = oItem.getBindingContext();
        if (oContext) {
          const oProject = oContext.getObject();
          aData.push(this._prepareExportItem(oProject));
        }
      });
            
      return aData;
    },
        
    /**
         * Prepare single item for export
         */
    _prepareExportItem: function (oProject) {
      return {
        name: oProject.name || '',
        description: oProject.description || '',
        status: oProject.status || '',
        priority: oProject.priority || '',
        department: oProject.department || '',
        projectManager: oProject.projectManager ? oProject.projectManager.displayName : '',
        startDate: oProject.startDate ? new Date(oProject.startDate) : null,
        endDate: oProject.endDate ? new Date(oProject.endDate) : null,
        budget: parseFloat(oProject.budget) || 0,
        currency: oProject.currency || 'USD',
        agentCount: oProject.agents ? oProject.agents.length : 0,
        lastDeployment: oProject.lastDeployment ? new Date(oProject.lastDeployment) : null,
        createdBy: oProject.createdBy || '',
        createdDate: oProject.createdDate ? new Date(oProject.createdDate) : null,
        modifiedBy: oProject.modifiedBy || '',
        modifiedDate: oProject.modifiedDate ? new Date(oProject.modifiedDate) : null
      };
    },
        
    /**
         * Verify data integrity before export
         */
    onVerifyDataIntegrity: function () {
      MessageToast.show('Verifying data integrity...');
            
      // Initialize integrity check model
      if (!this._oIntegrityModel) {
        this._oIntegrityModel = new JSONModel({
          inProgress: true,
          totalRecords: 0,
          checkedRecords: 0,
          errors: [],
          warnings: [],
          checksum: '',
          exportReady: false
        });
        this.setModel(this._oIntegrityModel, 'integrity');
      }
            
      // Open integrity check dialog
      if (!this._oIntegrityDialog) {
        Fragment.load({
          id: this.getView().getId(),
          name: 'sap.a2a.view.fragments.DataIntegrityDialog',
          controller: this
        }).then((oDialog) => {
          this._oIntegrityDialog = oDialog;
          this.getView().addDependent(this._oIntegrityDialog);
          this._oIntegrityDialog.open();
          this._performIntegrityCheck();
        });
      } else {
        this._oIntegrityDialog.open();
        this._performIntegrityCheck();
      }
    },
        
    /**
         * Perform comprehensive data integrity check
         */
    _performIntegrityCheck: function () {
      const oModel = this._oIntegrityModel;
      oModel.setProperty('/inProgress', true);
      oModel.setProperty('/errors', []);
      oModel.setProperty('/warnings', []);
            
      // Get all data
      const aData = this._getExportData('all');
      const aColumns = this._getExportColumns();
            
      oModel.setProperty('/totalRecords', aData.length);
            
      let nChecked = 0;
      const aErrors = [];
      const aWarnings = [];
      const aChecksums = [];
            
      // Process data in chunks for performance
      const processChunk = (startIndex) => {
        const chunkSize = 100;
        const endIndex = Math.min(startIndex + chunkSize, aData.length);
                
        for (let i = startIndex; i < endIndex; i++) {
          const oRecord = aData[i];
          const recordErrors = this._checkRecordIntegrity(oRecord, aColumns, i);
                    
          aErrors.push(...recordErrors.errors);
          aWarnings.push(...recordErrors.warnings);
          aChecksums.push(recordErrors.checksum);
                    
          nChecked++;
          oModel.setProperty('/checkedRecords', nChecked);
        }
                
        if (endIndex < aData.length) {
          // Continue with next chunk
          setTimeout(() => processChunk(endIndex), 10);
        } else {
          // All checks complete
          this._finalizeIntegrityCheck(aErrors, aWarnings, aChecksums);
        }
      };
            
      // Start processing
      processChunk(0);
    },
        
    /**
         * Check individual record integrity
         */
    _checkRecordIntegrity: function (oRecord, aColumns, nIndex) {
      const errors = [];
      const warnings = [];
      let recordData = '';
            
      aColumns.forEach(oCol => {
        const value = oRecord[oCol.property];
                
        // Check for required fields
        if (oCol.required && !value) {
          errors.push({
            row: nIndex + 1,
            column: oCol.label,
            type: 'Missing Required Field',
            value: 'Empty'
          });
        }
                
        // Check data types
        if (value !== null && value !== undefined) {
          if (oCol.type === 'number' && isNaN(parseFloat(value))) {
            warnings.push({
              row: nIndex + 1,
              column: oCol.label,
              type: 'Invalid Number Format',
              value: String(value)
            });
          }
                    
          if (oCol.type === 'date' && !(value instanceof Date)) {
            warnings.push({
              row: nIndex + 1,
              column: oCol.label,
              type: 'Invalid Date Format',
              value: String(value)
            });
          }
                    
          // Check for data truncation
          if (oCol.maxLength && String(value).length > oCol.maxLength) {
            warnings.push({
              row: nIndex + 1,
              column: oCol.label,
              type: 'Data Truncation Risk',
              value: `Length: ${String(value).length} (max: ${oCol.maxLength})`
            });
          }
        }
                
        // Build record data for checksum
        recordData += `${String(value || '')  }|`;
      });
            
      // Calculate simple checksum for record
      const checksum = this._calculateChecksum(recordData);
            
      return {
        errors: errors,
        warnings: warnings,
        checksum: checksum
      };
    },
        
    /**
         * Calculate checksum for data
         */
    _calculateChecksum: function (sData) {
      let hash = 0;
      for (let i = 0; i < sData.length; i++) {
        const char = sData.charCodeAt(i);
        hash = ((hash << 5) - hash) + char;
        hash = hash & hash; // Convert to 32-bit integer
      }
      return Math.abs(hash).toString(16);
    },
        
    /**
         * Finalize integrity check
         */
    _finalizeIntegrityCheck: function (aErrors, aWarnings, aChecksums) {
      const oModel = this._oIntegrityModel;
            
      // Calculate overall checksum
      const overallChecksum = this._calculateChecksum(aChecksums.join(''));
            
      oModel.setProperty('/errors', aErrors);
      oModel.setProperty('/warnings', aWarnings);
      oModel.setProperty('/checksum', overallChecksum);
      oModel.setProperty('/inProgress', false);
      oModel.setProperty('/exportReady', aErrors.length === 0);
            
      // Show summary
      if (aErrors.length === 0 && aWarnings.length === 0) {
        MessageToast.show('Data integrity check passed successfully!');
      } else if (aErrors.length > 0) {
        MessageBox.error(`Data integrity check failed with ${aErrors.length} errors.`);
      } else {
        MessageBox.warning(`Data integrity check completed with ${aWarnings.length} warnings.`);
      }
    },
        
    /**
         * Export with integrity verification
         */
    onExportWithIntegrity: function (sFormat) {
      // First verify integrity
      const aData = this._getExportData('visible');
      const aColumns = this._getExportColumns();
            
      // Quick integrity check
      let hasErrors = false;
      aData.forEach((oRecord, _index) => {
        aColumns.forEach(oCol => {
          if (oCol.required && !oRecord[oCol.property]) {
            hasErrors = true;
          }
        });
      });
            
      if (hasErrors) {
        MessageBox.confirm(
          'Data integrity issues detected. Do you want to continue with the export?',
          {
            title: 'Data Integrity Warning',
            actions: [MessageBox.Action.YES, MessageBox.Action.NO],
            onClose: function (oAction) {
              if (oAction === MessageBox.Action.YES) {
                this._proceedWithExport(sFormat);
              }
            }.bind(this)
          }
        );
      } else {
        this._proceedWithExport(sFormat);
      }
    },
        
    /**
         * Proceed with export after integrity check
         */
    _proceedWithExport: function (sFormat) {
      switch (sFormat) {
      case 'excel':
        this.onExportExcel();
        break;
      case 'pdf':
        this.onExportPDF();
        break;
      case 'csv':
        this.onExportCSV();
        break;
      default:
        MessageBox.error(`Unknown export format: ${  sFormat}`);
      }
    },
        
    /**
         * Save integrity report
         */
    onSaveIntegrityReport: function () {
      const oModel = this._oIntegrityModel;
      const oData = oModel.getData();
            
      // Generate report content
      let sReport = 'DATA INTEGRITY REPORT\n';
      sReport += '=====================\n\n';
      sReport += `Generated: ${  new Date().toLocaleString()  }\n`;
      sReport += `Total Records: ${  oData.totalRecords  }\n`;
      sReport += `Data Checksum: ${  oData.checksum  }\n`;
      sReport += `Errors Found: ${  oData.errors.length  }\n`;
      sReport += `Warnings Found: ${  oData.warnings.length  }\n\n`;
            
      if (oData.errors.length > 0) {
        sReport += 'ERRORS\n';
        sReport += '------\n';
        oData.errors.forEach((error, index) => {
          sReport += `${index + 1}. Row ${error.row}, Column "${error.column}"\n`;
          sReport += `   Type: ${error.type}\n`;
          sReport += `   Value: ${error.value}\n\n`;
        });
      }
            
      if (oData.warnings.length > 0) {
        sReport += '\nWARNINGS\n';
        sReport += '--------\n';
        oData.warnings.forEach((warning, index) => {
          sReport += `${index + 1}. Row ${warning.row}, Column "${warning.column}"\n`;
          sReport += `   Type: ${warning.type}\n`;
          sReport += `   Value: ${warning.value}\n\n`;
        });
      }
            
      // Create and download report
      const blob = new Blob([sReport], { type: 'text/plain;charset=utf-8;' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.setAttribute('href', url);
      link.setAttribute('download', `data_integrity_report_${  new Date().toISOString().slice(0, 10)  }.txt`);
      link.style.visibility = 'hidden';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
            
      MessageToast.show('Integrity report saved successfully');
    },
        
    /**
         * Close integrity dialog
         */
    onCloseIntegrityDialog: function () {
      if (this._oIntegrityDialog) {
        this._oIntegrityDialog.close();
      }
    },
        
    /**
         * Generate CSV content
         */
    _generateCSV: function (aColumns, aData) {
      let sCSV = '';
            
      // Add headers
      sCSV += `${aColumns.map(col => `"${  col.label  }"`).join(',')  }\n`;
            
      // Add data rows
      aData.forEach(oRow => {
        const aRowData = [];
        aColumns.forEach(oCol => {
          let value = oRow[oCol.property] || '';
                    
          // Format dates
          if (value instanceof Date) {
            value = value.toLocaleDateString();
          }
                    
          // Escape quotes and wrap in quotes
          if (typeof value === 'string') {
            value = `"${  value.replace(/"/g, '""')  }"`;
          }
                    
          aRowData.push(value);
        });
        sCSV += `${aRowData.join(',')  }\n`;
      });
            
      return sCSV;
    }
  });
});