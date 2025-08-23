sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/format/DateFormat"
], (Controller, JSONModel, MessageToast, MessageBox, DateFormat) => {
    "use strict";

    return Controller.extend("a2a.portal.controller.Projects", {

        onInit: function () {
            // Initialize view model
            const oViewModel = new JSONModel({
                viewMode: "tiles",
                projects: [],
                busy: false
            });
            this.getView().setModel(oViewModel, "view");

            // Load projects data
            this._loadProjects();
        },

        _loadProjects: function () {
            const oViewModel = this.getView().getModel("view");
            oViewModel.setProperty("/busy", true);

            jQuery.ajax({
                url: "/api/projects",
                method: "GET",
                success: function (data) {
                    oViewModel.setProperty("/projects", data.projects || []);
                    oViewModel.setProperty("/busy", false);
                }.bind(this),
                error: function (xhr, status, error) {
                    MessageToast.show(`Failed to load projects: ${  error}`);
                    oViewModel.setProperty("/busy", false);
                }.bind(this)
            });
        },

        onCreateProject: function () {
            if (!this._oCreateDialog) {
                this._oCreateDialog = sap.ui.xmlfragment("a2a.portal.fragment.CreateProjectDialog", this);
                this.getView().addDependent(this._oCreateDialog);
            }
            this._oCreateDialog.open();
        },

        onCreateProjectConfirm: function (oEvent) {
            const oDialog = oEvent.getSource().getParent();
            const sName = sap.ui.getCore().byId("createProjectName").getValue();
            const sDescription = sap.ui.getCore().byId("createProjectDescription").getValue();

            if (!sName.trim()) {
                MessageToast.show("Please enter a project name");
                return;
            }

            const oProjectData = {
                name: sName.trim(),
                description: sDescription.trim()
            };

            jQuery.ajax({
                url: "/api/projects",
                method: "POST",
                contentType: "application/json",
                data: JSON.stringify(oProjectData),
                success: function (_data) {
                    MessageToast.show("Project created successfully");
                    this._loadProjects();
                    oDialog.close();
                }.bind(this),
                error: function (xhr, _status, _error) {
                    let sMessage = "Failed to create project";
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
            MessageToast.show("Import functionality - coming soon");
            oDialog.close();
        },

        onImportProjectCancel: function (oEvent) {
            oEvent.getSource().getParent().close();
        },

        onImportProject: function () {
            if (!this._oImportDialog) {
                this._oImportDialog = sap.ui.xmlfragment("a2a.portal.fragment.ImportProjectDialog", this);
                this.getView().addDependent(this._oImportDialog);
            }
            this._oImportDialog.open();
        },

        onRefresh: function () {
            this._loadProjects();
            MessageToast.show("Projects refreshed");
        },

        onProjectPress: function (oEvent) {
            const oContext = oEvent.getSource().getBindingContext("view");
            const sProjectId = oContext.getProperty("project_id");
            
            // For now, just show a message since we're not using routing
            MessageToast.show(`Project selected: ${  sProjectId}`);
        },

        onViewChange: function (oEvent) {
            const sSelectedKey = oEvent.getParameter("item").getKey();
            const oViewModel = this.getView().getModel("view");
            oViewModel.setProperty("/viewMode", sSelectedKey);
        },

        onSearch: function (oEvent) {
            const sQuery = oEvent.getParameter("query");
            const oTable = this.byId("projectsTable");
            const oBinding = oTable.getBinding("items");

            if (sQuery && sQuery.length > 0) {
                const oFilter = new sap.ui.model.Filter([
                    new sap.ui.model.Filter("name", sap.ui.model.FilterOperator.Contains, sQuery),
                    new sap.ui.model.Filter("description", sap.ui.model.FilterOperator.Contains, sQuery),
                    new sap.ui.model.Filter("project_id", sap.ui.model.FilterOperator.Contains, sQuery)
                ], false);
                oBinding.filter([oFilter]);
            } else {
                oBinding.filter([]);
            }
        },

        onOpenFilterDialog: function () {
            MessageToast.show("Filter dialog - coming soon");
        },

        onOpenSortDialog: function () {
            if (!this._oSortDialog) {
                this._oSortDialog = sap.ui.xmlfragment("a2a.portal.fragment.SortDialog", this);
                this.getView().addDependent(this._oSortDialog);
            }
            this._oSortDialog.open();
        },

        onSortConfirm: function (oEvent) {
            const oSortItem = oEvent.getParameter("sortItem");
            const bDescending = oEvent.getParameter("sortDescending");
            const oTable = this.byId("projectsTable");
            const oBinding = oTable.getBinding("items");
            
            if (oSortItem) {
                const sSortPath = oSortItem.getKey();
                const oSorter = new sap.ui.model.Sorter(sSortPath, bDescending);
                oBinding.sort(oSorter);
            }
        },

        onEditProject: function (oEvent) {
            const oContext = oEvent.getSource().getBindingContext("view");
            const oProjectData = oContext.getProperty();
            
            if (!this._oEditDialog) {
                this._oEditDialog = sap.ui.xmlfragment("a2a.portal.fragment.EditProjectDialog", this);
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
            const oContext = oEvent.getSource().getBindingContext("view");
            const sProjectName = oContext.getProperty("name");
            
            MessageBox.confirm(
                `Clone project '${  sProjectName  }'?`, {
                    onClose: function (sAction) {
                        if (sAction === MessageBox.Action.OK) {
                            // Implement clone logic
                            MessageToast.show("Clone functionality - coming soon");
                        }
                    }
                }
            );
        },

        onDeleteProject: function (oEvent) {
            const oContext = oEvent.getSource().getBindingContext("view");
            const sProjectName = oContext.getProperty("name");
            const sProjectId = oContext.getProperty("project_id");
            
            MessageBox.confirm(
                `Delete project '${  sProjectName  }'? This action cannot be undone.`, {
                    icon: MessageBox.Icon.WARNING,
                    title: "Confirm Deletion",
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
                method: "DELETE",
                success: function () {
                    MessageToast.show("Project deleted successfully");
                    this._loadProjects();
                }.bind(this),
                error: function (xhr, status, error) {
                    MessageToast.show(`Failed to delete project: ${  error}`);
                }.bind(this)
            });
        },

        onExportSelected: function () {
            const oTable = this.byId("projectsTable");
            const aSelectedItems = oTable.getSelectedItems();
            
            if (aSelectedItems.length === 0) {
                MessageToast.show("Please select projects to export");
                return;
            }
            
            MessageToast.show("Export functionality - coming soon");
        },

        onDeleteSelected: function () {
            const oTable = this.byId("projectsTable");
            const aSelectedItems = oTable.getSelectedItems();
            
            if (aSelectedItems.length === 0) {
                MessageToast.show("Please select projects to delete");
                return;
            }
            
            const sMessage = `Delete ${  aSelectedItems.length  } selected project(s)? This action cannot be undone.`;
            
            MessageBox.confirm(sMessage, {
                icon: MessageBox.Icon.WARNING,
                title: "Confirm Deletion",
                onClose: function (sAction) {
                    if (sAction === MessageBox.Action.OK) {
                        // Implement batch delete
                        MessageToast.show("Batch delete functionality - coming soon");
                    }
                }
            });
        },

        onCreateFromTemplate: function () {
            MessageToast.show("Create from template - coming soon");
        },

        onViewTutorials: function () {
            MessageToast.show("Opening tutorials...");
        },

        // Formatters
        formatDate: function (sDate) {
            if (!sDate) {
                return "";
            }
            
            const oDateFormat = DateFormat.getDateTimeInstance({
                style: "medium"
            });
            
            return oDateFormat.format(new Date(sDate));
        },

        formatStatusState: function (sStatus) {
            switch (sStatus) {
                case "active": return "Success";
                case "inactive": return "Warning";
                case "error": return "Error";
                default: return "None";
            }
        }
    });
});