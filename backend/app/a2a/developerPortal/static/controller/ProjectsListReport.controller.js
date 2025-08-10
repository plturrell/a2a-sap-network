sap.ui.define([
    "sap/a2a/controller/BaseController",
    "sap/ui/model/json/JSONModel",
    "sap/ui/model/Filter",
    "sap/ui/model/FilterOperator",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Fragment",
    "sap/ui/export/Spreadsheet",
    "sap/ui/export/library"
], function (BaseController, JSONModel, Filter, FilterOperator, MessageToast, MessageBox, Fragment, Spreadsheet, exportLibrary) {
    "use strict";

    /**
     * SAP Fiori Elements List Report Controller for Projects
     * Implements enterprise patterns with smart controls and responsive design
     */
    return BaseController.extend("sap.a2a.controller.ProjectsListReport", {

        /* =========================================================== */
        /* lifecycle methods                                           */
        /* =========================================================== */

        /**
         * Called when the controller is instantiated.
         */
        onInit: function () {
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
                viewMode: "table",
                filterExpanded: false
            });
            
            this.setModel(this._oViewModel, "listReport");
            
            // Initialize filter and sort state
            this._mFilters = {
                search: [],
                status: [],
                priority: [],
                businessUnit: []
            };
            
            // Setup router event handling
            this.getRouter().getRoute("projects").attachPatternMatched(this._onPatternMatched, this);
            
            // Load initial data
            this._loadKPIData();
            this._loadProjectStatuses();
            this._loadProjectPriorities();
            
            // Setup help system integration
            this._setupContextualHelp();
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
            this.getRouter().navTo("projectCreate");
        },

        /**
         * Handles project item press - navigation to object page
         */
        onProjectPress: function (oEvent) {
            const oBindingContext = oEvent.getSource().getBindingContext();
            const sProjectId = oBindingContext.getProperty("id");
            
            this.getRouter().navTo("projectDetail", {
                projectId: sProjectId
            });
        },

        /**
         * Handles refresh button press
         */
        onRefresh: function () {
            this._refreshData();
            this._loadKPIData();
            MessageToast.show(this.getResourceBundle().getText("common.refreshed"));
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
            const aSelectedProjects = aSelectedItems.map(function (oItem) {
                return oItem.getBindingContext().getObject();
            });
            
            this._oViewModel.setProperty("/selectedProjects", aSelectedProjects);
            
            // Update bulk action button states
            this._updateBulkActionButtons(aSelectedProjects.length > 0);
        },

        /**
         * Handles before rebind table event for smart table
         */
        onBeforeRebindTable: function (oEvent) {
            const mBindingParams = oEvent.getParameter("bindingParams");
            
            // Add custom filters
            Object.keys(this._mFilters).forEach(sFilterKey => {
                if (this._mFilters[sFilterKey].length > 0) {
                    mBindingParams.filters = mBindingParams.filters.concat(this._mFilters[sFilterKey]);
                }
            });
            
            // Add default sorting
            if (!mBindingParams.sorter || mBindingParams.sorter.length === 0) {
                mBindingParams.sorter = [
                    new sap.ui.model.Sorter("modifiedAt", true) // Sort by modified date, descending
                ];
            }
        },

        /**
         * Handles KPI tile press
         */
        onKPIPress: function (oEvent) {
            const sTileId = oEvent.getSource().getId();
            let sFilter = "";
            
            if (sTileId.includes("active")) {
                sFilter = "status eq 'ACTIVE'";
            } else if (sTileId.includes("completed")) {
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
            const sSelectedKey = oEvent.getParameter("key");
            this._oViewModel.setProperty("/viewMode", sSelectedKey);
            
            if (sSelectedKey === "cards") {
                this._switchToCardView();
            } else {
                this._switchToTableView();
            }
        },

        /**
         * Handles bulk edit operation
         */
        onBulkEdit: function () {
            const aSelectedProjects = this._oViewModel.getProperty("/selectedProjects");
            
            if (aSelectedProjects.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("projects.noSelection"));
                return;
            }
            
            // Open bulk edit dialog
            this._openBulkEditDialog(aSelectedProjects);
        },

        /**
         * Handles bulk delete operation
         */
        onBulkDelete: function () {
            const aSelectedProjects = this._oViewModel.getProperty("/selectedProjects");
            
            if (aSelectedProjects.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("projects.noSelection"));
                return;
            }
            
            const sConfirmText = this.getResourceBundle().getText("projects.deleteConfirm", [aSelectedProjects.length]);
            
            MessageBox.confirm(sConfirmText, {
                title: this.getResourceBundle().getText("projects.deleteTitle"),
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
            const sProjectId = oBindingContext.getProperty("id");
            
            this.getRouter().navTo("projectEdit", {
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
            
            const sConfirmText = this.getResourceBundle().getText("projects.deleteProjectConfirm", [oProject.name]);
            
            MessageBox.confirm(sConfirmText, {
                title: this.getResourceBundle().getText("projects.deleteTitle"),
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
                    name: "sap.a2a.view.fragments.ImportProjectsDialog",
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
            this.getRouter().navTo("templates");
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
                    name: "sap.a2a.view.fragments.ListReportSettingsDialog",
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
            helpProvider.showDetailedHelp(this.getView(), "projects_list_report", {
                title: this.getResourceBundle().getText("help.projects.title"),
                placement: "PreferredBottomOrFlip"
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
                "DRAFT": "None",
                "ACTIVE": "Success",
                "TESTING": "Warning",
                "DEPLOYED": "Success",
                "PAUSED": "Warning",
                "COMPLETED": "Success",
                "ARCHIVED": "None",
                "ERROR": "Error"
            };
            
            return mStatusStates[sStatus] || "None";
        },

        /**
         * Formats status to icon
         */
        formatStatusIcon: function (sStatus) {
            const mStatusIcons = {
                "DRAFT": "sap-icon://document",
                "ACTIVE": "sap-icon://play",
                "TESTING": "sap-icon://lab",
                "DEPLOYED": "sap-icon://cloud",
                "PAUSED": "sap-icon://pause",
                "COMPLETED": "sap-icon://complete",
                "ARCHIVED": "sap-icon://archive",
                "ERROR": "sap-icon://error"
            };
            
            return mStatusIcons[sStatus] || "sap-icon://question-mark";
        },

        /**
         * Formats priority to state for ObjectStatus
         */
        formatPriorityState: function (sPriority) {
            const mPriorityStates = {
                "LOW": "None",
                "MEDIUM": "Warning",
                "HIGH": "Error",
                "CRITICAL": "Error"
            };
            
            return mPriorityStates[sPriority] || "None";
        },

        /**
         * Formats priority to icon
         */
        formatPriorityIcon: function (sPriority) {
            const mPriorityIcons = {
                "LOW": "sap-icon://down",
                "MEDIUM": "sap-icon://right",
                "HIGH": "sap-icon://up",
                "CRITICAL": "sap-icon://alert"
            };
            
            return mPriorityIcons[sPriority];
        },

        /**
         * Formats priority to highlight for list items
         */
        formatPriorityHighlight: function (sPriority) {
            const mPriorityHighlights = {
                "CRITICAL": "Error",
                "HIGH": "Warning",
                "MEDIUM": "Information",
                "LOW": "None"
            };
            
            return mPriorityHighlights[sPriority] || "None";
        },

        /**
         * Formats relative time for dates
         */
        formatRelativeTime: function (dDate) {
            if (!dDate) return this.getResourceBundle().getText("common.never");
            
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
            if (!sStatus) return "";
            
            const mStatusTexts = {
                "SUCCESS": this.getResourceBundle().getText("projects.deploymentSuccess"),
                "FAILED": this.getResourceBundle().getText("projects.deploymentFailed"),
                "IN_PROGRESS": this.getResourceBundle().getText("projects.deploymentInProgress")
            };
            
            return mStatusTexts[sStatus] || sStatus;
        },

        /**
         * Formats percentage values
         */
        formatPercentage: function (nValue) {
            if (nValue === null || nValue === undefined) return "0%";
            return Math.round(nValue) + "%";
        },

        /**
         * Determines if project is deployable
         */
        isDeployable: function (sStatus) {
            return sStatus === "DRAFT" || sStatus === "TESTING" || sStatus === "PAUSED";
        },

        /**
         * Determines if project is stoppable
         */
        isStoppable: function (sStatus) {
            return sStatus === "ACTIVE" || sStatus === "DEPLOYED";
        },

        /* =========================================================== */
        /* internal methods                                            */
        /* =========================================================== */

        /**
         * Loads KPI data from backend
         */
        _loadKPIData: function () {
            this._oViewModel.setProperty("/busy", true);
            
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
                { key: "DRAFT", text: this.getResourceBundle().getText("projects.status.draft") },
                { key: "ACTIVE", text: this.getResourceBundle().getText("projects.status.active") },
                { key: "TESTING", text: this.getResourceBundle().getText("projects.status.testing") },
                { key: "DEPLOYED", text: this.getResourceBundle().getText("projects.status.deployed") },
                { key: "PAUSED", text: this.getResourceBundle().getText("projects.status.paused") },
                { key: "COMPLETED", text: this.getResourceBundle().getText("projects.status.completed") },
                { key: "ARCHIVED", text: this.getResourceBundle().getText("projects.status.archived") }
            ];
            
            this.getModel().setProperty("/ProjectStatuses", aStatuses);
        },

        /**
         * Loads project priority options
         */
        _loadProjectPriorities: function () {
            const aPriorities = [
                { key: "LOW", text: this.getResourceBundle().getText("projects.priority.low") },
                { key: "MEDIUM", text: this.getResourceBundle().getText("projects.priority.medium") },
                { key: "HIGH", text: this.getResourceBundle().getText("projects.priority.high") },
                { key: "CRITICAL", text: this.getResourceBundle().getText("projects.priority.critical") }
            ];
            
            this.getModel().setProperty("/ProjectPriorities", aPriorities);
        },

        /**
         * Refreshes main data
         */
        _refreshData: function () {
            const oSmartTable = this.byId("projectsSmartTable");
            if (oSmartTable) {
                oSmartTable.rebindTable();
            }
        },

        /**
         * Applies current filters to the table
         */
        _applyFilters: function () {
            const oSmartTable = this.byId("projectsSmartTable");
            if (oSmartTable) {
                oSmartTable.rebindTable();
            }
        },

        /**
         * Applies quick filter
         */
        _applyQuickFilter: function (sFilter) {
            const oSmartTable = this.byId("projectsSmartTable");
            const oTable = oSmartTable.getTable();
            
            // Create filter from OData filter string
            const aFilters = [new Filter({
                path: sFilter.split(" ")[0],
                operator: FilterOperator.EQ,
                value1: sFilter.split("'")[1]
            })];
            
            oTable.getBinding("items").filter(aFilters);
        },

        /**
         * Updates bulk action button states
         */
        _updateBulkActionButtons: function (bEnabled) {
            this.byId("bulkEditButton").setEnabled(bEnabled);
            this.byId("bulkDeleteButton").setEnabled(bEnabled);
            
            const sSelectionText = bEnabled ? 
                this.getResourceBundle().getText("projects.selectedCount", [this._oViewModel.getProperty("/selectedProjects").length]) : 
                "";
            
            this.byId("selectionInfo").setText(sSelectionText).setVisible(bEnabled);
        },

        /**
         * Switches to card view layout
         */
        _switchToCardView: function () {
            // Implementation for card view layout
            MessageToast.show(this.getResourceBundle().getText("common.cardViewNotImplemented"));
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
                MessageToast.show(this.getResourceBundle().getText("projects.deploymentStarted", [oProject.name]));
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
                MessageToast.show(this.getResourceBundle().getText("projects.projectStopped", [oProject.name]));
                this._refreshData();
            }, 1000);
        },

        /**
         * Clones a project
         */
        _cloneProject: function (oProject) {
            this.getRouter().navTo("projectCreate", {}, {
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
                MessageToast.show(this.getResourceBundle().getText("projects.projectDeleted", [oProject.name]));
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
                MessageToast.show(this.getResourceBundle().getText("projects.projectsDeleted", [aProjects.length]));
                this._refreshData();
                this._loadKPIData();
                
                // Clear selection
                this._oViewModel.setProperty("/selectedProjects", []);
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
                    name: "sap.a2a.view.fragments.BulkEditProjectsDialog",
                    controller: this
                }).then((oDialog) => {
                    this.getView().addDependent(oDialog);
                    return oDialog;
                });
            }
            
            this._pBulkEditDialog.then((oDialog) => {
                oDialog.getModel("bulkEdit").setProperty("/selectedProjects", aProjects);
                oDialog.open();
            });
        },

        /**
         * Exports data to Excel
         */
        _exportToExcel: function () {
            const aColumns = [
                { property: "name", label: this.getResourceBundle().getText("projects.name") },
                { property: "status", label: this.getResourceBundle().getText("projects.status") },
                { property: "priority", label: this.getResourceBundle().getText("projects.priority") },
                { property: "createdBy", label: this.getResourceBundle().getText("projects.createdBy") },
                { property: "createdAt", label: this.getResourceBundle().getText("projects.createdAt"), type: exportLibrary.EdmType.DateTime },
                { property: "modifiedAt", label: this.getResourceBundle().getText("projects.modifiedAt"), type: exportLibrary.EdmType.DateTime }
            ];
            
            const oSpreadsheet = new Spreadsheet({
                workbook: {
                    columns: aColumns
                },
                dataSource: this.getModel().getProperty("/projects"),
                fileName: "A2A_Projects_Export"
            });
            
            oSpreadsheet.build();
        },

        /**
         * Exports single project data
         */
        _exportSingleProject: function (oProject) {
            // Implementation for single project export
            MessageToast.show(this.getResourceBundle().getText("projects.exportStarted", [oProject.name]));
        },

        /**
         * Opens business unit value help
         */
        _openBusinessUnitValueHelp: function (oSource) {
            // Implementation for business unit value help
            MessageToast.show(this.getResourceBundle().getText("common.valueHelpNotImplemented"));
        },

        /**
         * Sets up contextual help system
         */
        _setupContextualHelp: function () {
            const helpProvider = sap.a2a.utils.HelpProvider;
            
            // Enable help for key controls
            helpProvider.enableHelp(this.byId("createProjectButton"), "projects.create");
            helpProvider.enableHelp(this.byId("listReportFilter"), "projects.filter");
            helpProvider.enableHelp(this.byId("projectsSmartTable"), "projects.table");
        }
    });
});