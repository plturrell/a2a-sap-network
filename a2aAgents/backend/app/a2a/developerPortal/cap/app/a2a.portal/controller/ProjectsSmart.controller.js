sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/ui/model/Filter",
    "sap/ui/model/FilterOperator",
    "sap/ui/model/Sorter",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Fragment",
    "sap/ui/export/library",
    "sap/ui/export/Spreadsheet"
], (Controller, JSONModel, Filter, FilterOperator, Sorter, MessageToast, MessageBox, Fragment, exportLibrary, Spreadsheet) => {
    "use strict";

    const EdmType = exportLibrary.EdmType;

    return Controller.extend("com.sap.a2a.developerportal.controller.ProjectsSmart", {

        onInit: function () {
            this.oRouter = this.getOwnerComponent().getRouter();
            
            // Initialize models
            this._initializeModels();
            
            // Load projects data
            this._loadProjects();
            
            // Load statistics
            this._loadStatistics();
        },

        _initializeModels: function () {
            // Projects model
            const oProjectsModel = new JSONModel({
                projects: [],
                statistics: {
                    active_projects: 0,
                    deployed_projects: 0,
                    total_agents: 0,
                    avg_performance: 0
                }
            });
            this.getView().setModel(oProjectsModel);
            
            // Filter model for SmartFilterBar
            const oFilterModel = new JSONModel({
                status: "",
                type: "",
                created_date: null,
                name: ""
            });
            this.getView().setModel(oFilterModel, "filter");
        },

        _loadProjects: function () {
            const that = this;
            
            jQuery.ajax({
                url: "/api/v2/projects",
                method: "GET",
                success: function (oData) {
                    const oModel = that.getView().getModel();
                    oModel.setProperty("/projects", oData.projects || []);
                    
                    // Update table binding
                    that._refreshTable();
                },
                error: function (oError) {
                    console.error("Failed to load projects:", oError);
                    MessageToast.show("Failed to load projects");
                }
            });
        },

        _loadStatistics: function () {
            const that = this;
            
            jQuery.ajax({
                url: "/api/v2/projects/statistics",
                method: "GET",
                success: function (oData) {
                    const oModel = that.getView().getModel();
                    oModel.setProperty("/statistics", oData);
                },
                error: function (oError) {
                    console.error("Failed to load statistics:", oError);
                }
            });
        },

        _refreshTable: function () {
            const oTable = this.byId("projectsTable");
            if (oTable) {
                oTable.getBinding("items").refresh();
            }
        },

        // SmartTable Events
        onSmartTableInitialised: function (oEvent) {
            const oSmartTable = oEvent.getSource();
            const oTable = oSmartTable.getTable();
            
            // Configure table settings
            oTable.setGrowingThreshold(50);
            oTable.setSticky("ColumnHeaders,HeaderToolbar");
        },

        onBeforeRebindTable: function (oEvent) {
            const oBindingParams = oEvent.getParameter("bindingParams");
            const oSmartFilterBar = this.byId("smartFilterBar");
            
            if (oSmartFilterBar) {
                const aFilters = oSmartFilterBar.getFilters();
                oBindingParams.filters = aFilters;
            }
        },

        // SmartFilterBar Events
        onSearch: function (_oEvent) {
            const oSmartTable = this.byId("smartTable");
            if (oSmartTable) {
                oSmartTable.rebindTable();
            }
        },

        onFilterChange: function (oEvent) {
            // Handle filter changes
            // eslint-disable-next-line no-console
            // eslint-disable-next-line no-console
            console.log("Filter changed:", oEvent.getParameter("reason"));
        },

        // Quick Search
        onQuickSearch: function (oEvent) {
            const sQuery = oEvent.getParameter("newValue");
            const oTable = this.byId("projectsTable");
            const oBinding = oTable.getBinding("items");
            
            if (sQuery && sQuery.length > 0) {
                const aFilters = [
                    new Filter("name", FilterOperator.Contains, sQuery),
                    new Filter("description", FilterOperator.Contains, sQuery),
                    new Filter("type", FilterOperator.Contains, sQuery)
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

        // Project Actions
        onCreateProject: function () {
            this.oRouter.navTo("projectCreate");
        },

        onProjectPress: function (oEvent) {
            const oBindingContext = oEvent.getSource().getBindingContext();
            const sProjectId = oBindingContext.getProperty("project_id");
            
            this.oRouter.navTo("projectDetail", {
                projectId: sProjectId
            });
        },

        onEditProject: function (oEvent) {
            const oBindingContext = oEvent.getSource().getBindingContext();
            const sProjectId = oBindingContext.getProperty("project_id");
            
            this.oRouter.navTo("projectEdit", {
                projectId: sProjectId
            });
        },

        onDeployProject: function (oEvent) {
            const oBindingContext = oEvent.getSource().getBindingContext();
            const oProject = oBindingContext.getObject();
            
            MessageBox.confirm(
                `Are you sure you want to deploy project '${  oProject.name  }'?`,
                {
                    title: "Deploy Project",
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
                method: "POST",
                success: function () {
                    MessageToast.show("Project deployment started");
                    that._loadProjects(); // Refresh data
                },
                error: function (oError) {
                    console.error("Failed to deploy project:", oError);
                    MessageToast.show("Failed to deploy project");
                }
            });
        },

        onMoreActions: function (oEvent) {
            const oButton = oEvent.getSource();
            const oBindingContext = oButton.getBindingContext();
            
            if (!this._oActionsPopover) {
                Fragment.load({
                    name: "com.sap.a2a.developerportal.view.fragments.ProjectActionsPopover",
                    controller: this
                }).then((oPopover) => {
                    this._oActionsPopover = oPopover;
                    this.getView().addDependent(oPopover);
                    oPopover.setBindingContext(oBindingContext);
                    oPopover.openBy(oButton);
                });
            } else {
                this._oActionsPopover.setBindingContext(oBindingContext);
                this._oActionsPopover.openBy(oButton);
            }
        },

        // Toolbar Actions
        onImportProject: function () {
            MessageToast.show("Import project functionality");
        },

        onExportSelected: function () {
            const oTable = this.byId("projectsTable");
            const aSelectedItems = oTable.getSelectedItems();
            
            if (aSelectedItems.length === 0) {
                MessageToast.show("Please select projects to export");
                return;
            }
            
            this._exportProjects(aSelectedItems);
        },

        _exportProjects: function (aSelectedItems) {
            const aData = [];
            
            aSelectedItems.forEach((oItem) => {
                const oContext = oItem.getBindingContext();
                const oProject = oContext.getObject();
                
                aData.push({
                    name: oProject.name,
                    status: oProject.status,
                    type: oProject.type,
                    created_date: oProject.created_date,
                    agents_count: oProject.agents_count,
                    deployment_status: oProject.deployment_status
                });
            });
            
            const aCols = [
                { label: "Project Name", property: "name", type: EdmType.String },
                { label: "Status", property: "status", type: EdmType.String },
                { label: "Type", property: "type", type: EdmType.String },
                { label: "Created Date", property: "created_date", type: EdmType.DateTime },
                { label: "Agents Count", property: "agents_count", type: EdmType.Number },
                { label: "Deployment Status", property: "deployment_status", type: EdmType.String }
            ];
            
            const oSettings = {
                workbook: {
                    columns: aCols,
                    hierarchyLevel: 'Level'
                },
                dataSource: aData,
                fileName: "A2A_Projects_Export.xlsx",
                worker: false
            };
            
            const oSheet = new Spreadsheet(oSettings);
            oSheet.build().finally(() => {
                oSheet.destroy();
            });
        },

        onRefresh: function () {
            this._loadProjects();
            this._loadStatistics();
            MessageToast.show("Data refreshed");
        },

        // Statistics Cards Actions
        onFilterByStatus: function (oEvent) {
            const sStatus = oEvent.getSource().data("status");
            const oSmartFilterBar = this.byId("smartFilterBar");
            
            if (oSmartFilterBar) {
                // Set filter value
                const oFilterModel = this.getView().getModel("filter");
                oFilterModel.setProperty("/status", sStatus);
                
                // Trigger search
                oSmartFilterBar.search();
            }
        },

        onShowAgentsOverview: function () {
            this.oRouter.navTo("agentsOverview");
        },

        onShowPerformanceDetails: function () {
            this.oRouter.navTo("performanceDashboard");
        },

        // Formatters
        formatDate: function (sDate) {
            if (!sDate) {
return "";
}
            
            const oDate = new Date(sDate);
            return oDate.toLocaleDateString();
        },

        formatRelativeTime: function (sDate) {
            if (!sDate) {
return "";
}
            
            const oDate = new Date(sDate);
            const oNow = new Date();
            const iDiff = oNow.getTime() - oDate.getTime();
            const iDays = Math.floor(iDiff / (1000 * 60 * 60 * 24));
            
            if (iDays === 0) {
                return "Today";
            } else if (iDays === 1) {
                return "Yesterday";
            } else if (iDays < 7) {
                return `${iDays  } days ago`;
            } else {
                return oDate.toLocaleDateString();
            }
        },

        formatStatusState: function (sStatus) {
            switch (sStatus) {
                case "active":
                    return "Success";
                case "deployed":
                    return "Success";
                case "inactive":
                    return "Warning";
                case "error":
                    return "Error";
                default:
                    return "None";
            }
        },

        formatStatusIcon: function (sStatus) {
            switch (sStatus) {
                case "active":
                    return "sap-icon://status-positive";
                case "deployed":
                    return "sap-icon://status-positive";
                case "inactive":
                    return "sap-icon://status-inactive";
                case "error":
                    return "sap-icon://status-negative";
                default:
                    return "";
            }
        },

        formatStatusColor: function (sStatus) {
            switch (sStatus) {
                case "active":
                    return "Positive";
                case "deployed":
                    return "Positive";
                case "inactive":
                    return "Critical";
                case "error":
                    return "Negative";
                default:
                    return "Neutral";
            }
        },

        formatAgentsState: function (iCount) {
            if (iCount > 10) {
                return "Success";
            } else if (iCount > 5) {
                return "Warning";
            } else {
                return "None";
            }
        },

        formatDeploymentState: function (sStatus) {
            switch (sStatus) {
                case "deployed":
                    return "Success";
                case "deploying":
                    return "Warning";
                case "failed":
                    return "Error";
                default:
                    return "None";
            }
        },

        formatDeploymentIcon: function (sStatus) {
            switch (sStatus) {
                case "deployed":
                    return "sap-icon://cloud";
                case "deploying":
                    return "sap-icon://pending";
                case "failed":
                    return "sap-icon://error";
                default:
                    return "";
            }
        },

        formatDeployEnabled: function (sStatus) {
            return sStatus === "active";
        }
    });
});
