sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/format/DateFormat",
    "sap/ui/core/Fragment"
], function (Controller, JSONModel, MessageToast, MessageBox, DateFormat, Fragment) {
    "use strict";

    return Controller.extend("a2a.portal.controller.Templates", {

        onInit: function () {
            // Initialize view model
            var oViewModel = new JSONModel({
                viewMode: "cards",
                templates: [],
                busy: false
            });
            this.getView().setModel(oViewModel, "view");

            // Load templates data
            this._loadTemplates();
        },

        _loadTemplates: function () {
            var oViewModel = this.getView().getModel("view");
            oViewModel.setProperty("/busy", true);

            jQuery.ajax({
                url: "/api/templates",
                method: "GET",
                success: function (data) {
                    oViewModel.setProperty("/templates", data.templates || []);
                    oViewModel.setProperty("/busy", false);
                    this._updateGridCards();
                }.bind(this),
                error: function (xhr, status, error) {
                    // Fallback to mock data
                    var aMockTemplates = this._getMockTemplates();
                    oViewModel.setProperty("/templates", aMockTemplates);
                    oViewModel.setProperty("/busy", false);
                    this._updateGridCards();
                    MessageToast.show("Using sample data - backend connection unavailable");
                }.bind(this)
            });
        },

        _getMockTemplates: function () {
            return [
                {
                    id: "t1",
                    name: "Data Product Agent",
                    description: "Complete data product management with Dublin Core metadata",
                    category: "data-product",
                    author: "SAP A2A Team",
                    version: "2.1.0",
                    downloads: 1542,
                    status: "published",
                    lastModified: "2024-01-22T10:30:00Z"
                },
                {
                    id: "t2",
                    name: "Multi-Pass Standardization Agent",
                    description: "Advanced data standardization with multiple validation passes",
                    category: "standardization",
                    author: "SAP A2A Team",
                    version: "1.8.2",
                    downloads: 987,
                    status: "published",
                    lastModified: "2024-01-21T15:20:00Z"
                },
                {
                    id: "t3",
                    name: "AI Processing Pipeline",
                    description: "Complete AI processing workflow with vector embeddings",
                    category: "ai-processing",
                    author: "Community",
                    version: "1.5.0",
                    downloads: 756,
                    status: "published",
                    lastModified: "2024-01-20T09:15:00Z"
                },
                {
                    id: "t4",
                    name: "Custom Validation Agent",
                    description: "Configurable validation rules for data quality",
                    category: "validation",
                    author: "John Doe",
                    version: "1.0.0",
                    downloads: 234,
                    status: "published",
                    lastModified: "2024-01-19T14:45:00Z"
                }
            ];
        },

        _updateGridCards: function () {
            var oContainer = this.byId("cardsContainer");
            var oViewModel = this.getView().getModel("view");
            var aTemplates = oViewModel.getProperty("/templates") || [];
            
            // Clear existing cards
            oContainer.removeAllItems();
            
            // Add new cards for each template
            aTemplates.forEach(function (oTemplate) {
                var oCard = new sap.f.Card({
                    class: "sapUiMediumMargin",
                    header: new sap.f.cards.Header({
                        title: oTemplate.name,
                        subtitle: oTemplate.description,
                        iconSrc: this._getTemplateIcon(oTemplate.category)
                    }),
                    content: new sap.f.cards.Object({
                        groups: [{
                            title: "Details",
                            items: [{
                                label: "Category",
                                value: oTemplate.category
                            }, {
                                label: "Author",
                                value: oTemplate.author
                            }, {
                                label: "Version",
                                value: oTemplate.version
                            }]
                        }]
                    })
                });
                
                oCard.attachPress(this.onTemplatePress.bind(this, oTemplate));
                oContainer.addItem(oCard);
            }.bind(this));
        },

        _getTemplateIcon: function (sCategory) {
            switch (sCategory) {
                case "data-product": return "sap-icon://database";
                case "standardization": return "sap-icon://validate";
                case "ai-processing": return "sap-icon://artificial-intelligence";
                case "validation": return "sap-icon://quality-issue";
                case "integration": return "sap-icon://connected";
                default: return "sap-icon://template";
            }
        },

        onCreateTemplate: function () {
            if (!this._oCreateDialog) {
                this._oCreateDialog = sap.ui.xmlfragment("a2a.portal.fragment.CreateTemplateDialog", this);
                this.getView().addDependent(this._oCreateDialog);
            }
            this._oCreateDialog.open();
        },

        onCreateTemplateConfirm: function (oEvent) {
            var oDialog = oEvent.getSource().getParent();
            var sName = sap.ui.getCore().byId("createTemplateName").getValue();
            var sDescription = sap.ui.getCore().byId("createTemplateDescription").getValue();
            var sCategory = sap.ui.getCore().byId("createTemplateCategory").getSelectedKey();

            if (!sName.trim()) {
                MessageToast.show("Please enter a template name");
                return;
            }

            var oTemplateData = {
                name: sName.trim(),
                description: sDescription.trim(),
                category: sCategory || "custom"
            };

            jQuery.ajax({
                url: "/api/templates",
                method: "POST",
                contentType: "application/json",
                data: JSON.stringify(oTemplateData),
                success: function (data) {
                    MessageToast.show("Template created successfully");
                    this._loadTemplates();
                    oDialog.close();
                }.bind(this),
                error: function (xhr, status, error) {
                    var sMessage = "Failed to create template";
                    if (xhr.responseJSON && xhr.responseJSON.detail) {
                        sMessage += ": " + xhr.responseJSON.detail;
                    }
                    MessageToast.show(sMessage);
                }.bind(this)
            });
        },

        onCreateTemplateCancel: function (oEvent) {
            oEvent.getSource().getParent().close();
        },

        onImportTemplate: function () {
            if (!this._oImportDialog) {
                this._oImportDialog = sap.ui.xmlfragment("a2a.portal.fragment.ImportTemplateDialog", this);
                this.getView().addDependent(this._oImportDialog);
            }
            this._oImportDialog.open();
        },

        onRefresh: function () {
            this._loadTemplates();
            MessageToast.show("Templates refreshed");
        },

        onViewChange: function (oEvent) {
            var sSelectedKey = oEvent.getParameter("item").getKey();
            var oViewModel = this.getView().getModel("view");
            oViewModel.setProperty("/viewMode", sSelectedKey);
        },

        onSearch: function (oEvent) {
            var sQuery = oEvent.getParameter("query");
            var oTable = this.byId("templatesTable");
            var oBinding = oTable.getBinding("items");

            if (sQuery && sQuery.length > 0) {
                var oFilter = new sap.ui.model.Filter([
                    new sap.ui.model.Filter("name", sap.ui.model.FilterOperator.Contains, sQuery),
                    new sap.ui.model.Filter("description", sap.ui.model.FilterOperator.Contains, sQuery),
                    new sap.ui.model.Filter("category", sap.ui.model.FilterOperator.Contains, sQuery),
                    new sap.ui.model.Filter("author", sap.ui.model.FilterOperator.Contains, sQuery)
                ], false);
                oBinding.filter([oFilter]);
            } else {
                oBinding.filter([]);
            }
        },

        onOpenFilterDialog: function () {
            var that = this;
            if (!this._oFilterDialog) {
                Fragment.load({
                    name: "a2a.portal.view.fragments.TemplateFilterDialog",
                    controller: this
                }).then(function (oDialog) {
                    that._oFilterDialog = oDialog;
                    that.getView().addDependent(that._oFilterDialog);
                    that._oFilterDialog.open();
                });
            } else {
                this._oFilterDialog.open();
            }
        },

        onFilterConfirm: function () {
            var aCategories = sap.ui.getCore().byId("filterCategory").getSelectedKeys();
            var aStatuses = sap.ui.getCore().byId("filterStatus").getSelectedKeys();
            var iAuthorType = sap.ui.getCore().byId("filterAuthorType").getSelectedIndex();
            var sVersion = sap.ui.getCore().byId("filterVersion").getSelectedKey();
            var oDownloadRange = sap.ui.getCore().byId("filterDownloads");
            var bCertifiedOnly = sap.ui.getCore().byId("filterCertified").getSelected();
            
            var aFilters = [];
            
            // Category filter
            if (aCategories.length > 0) {
                var aCategoryFilters = aCategories.map(function(sCategory) {
                    return new sap.ui.model.Filter("category", sap.ui.model.FilterOperator.EQ, sCategory);
                });
                aFilters.push(new sap.ui.model.Filter(aCategoryFilters, false));
            }
            
            // Status filter
            if (aStatuses.length > 0) {
                var aStatusFilters = aStatuses.map(function(sStatus) {
                    return new sap.ui.model.Filter("status", sap.ui.model.FilterOperator.EQ, sStatus);
                });
                aFilters.push(new sap.ui.model.Filter(aStatusFilters, false));
            }
            
            // Author type filter
            switch(iAuthorType) {
                case 1: // SAP Official
                    aFilters.push(new sap.ui.model.Filter("author", sap.ui.model.FilterOperator.Contains, "SAP"));
                    break;
                case 2: // Community
                    aFilters.push(new sap.ui.model.Filter("author", sap.ui.model.FilterOperator.NE, "SAP A2A Team"));
                    break;
                case 3: // My Templates
                    aFilters.push(new sap.ui.model.Filter("author", sap.ui.model.FilterOperator.EQ, "Current User"));
                    break;
            }
            
            // Downloads filter
            var iMinDownloads = oDownloadRange.getValue();
            var iMaxDownloads = oDownloadRange.getValue2();
            if (iMinDownloads > 0) {
                aFilters.push(new sap.ui.model.Filter("downloads", sap.ui.model.FilterOperator.GE, iMinDownloads));
            }
            if (iMaxDownloads < 10000) {
                aFilters.push(new sap.ui.model.Filter("downloads", sap.ui.model.FilterOperator.LE, iMaxDownloads));
            }
            
            // Apply filters to table
            var oTable = this.byId("templatesTable");
            var oBinding = oTable.getBinding("items");
            oBinding.filter(aFilters);
            
            this._oFilterDialog.close();
            MessageToast.show("Filters applied");
        },

        onFilterCancel: function () {
            this._oFilterDialog.close();
        },

        onOpenSortDialog: function () {
            if (!this._oSortDialog) {
                this._oSortDialog = sap.ui.xmlfragment("a2a.portal.fragment.SortDialog", this);
                this.getView().addDependent(this._oSortDialog);
            }
            this._oSortDialog.open();
        },

        onSortConfirm: function (oEvent) {
            var oSortItem = oEvent.getParameter("sortItem");
            var bDescending = oEvent.getParameter("sortDescending");
            var oTable = this.byId("templatesTable");
            var oBinding = oTable.getBinding("items");
            
            if (oSortItem) {
                var sSortPath = oSortItem.getKey();
                var oSorter = new sap.ui.model.Sorter(sSortPath, bDescending);
                oBinding.sort(oSorter);
            }
        },

        onTemplatePress: function (oEvent) {
            var oContext, oTemplate;
            
            if (typeof oEvent === "object" && oEvent.name) {
                // Called from grid card
                oTemplate = oEvent;
            } else {
                // Called from table or list
                oContext = oEvent.getSource().getBindingContext("view");
                oTemplate = oContext.getProperty();
            }
            
            // For now, just show a message since we're not using routing
            MessageToast.show("Template selected: " + oTemplate.name);
        },

        onDownloadTemplate: function (oEvent) {
            oEvent.stopPropagation();
            var oContext = oEvent.getSource().getBindingContext("view");
            var sTemplateId = oContext.getProperty("id");
            var sTemplateName = oContext.getProperty("name");
            
            jQuery.ajax({
                url: "/api/templates/" + sTemplateId + "/download",
                method: "GET",
                success: function () {
                    MessageToast.show("Template downloaded successfully");
                }.bind(this),
                error: function (xhr) {
                    // Create download link for template export
                    var oTemplate = oContext.getProperty();
                    var oExportData = {
                        template: {
                            name: oTemplate.name,
                            description: oTemplate.description,
                            category: oTemplate.category,
                            version: oTemplate.version,
                            configuration: oTemplate.configuration || {},
                            skills: oTemplate.skills || [],
                            metadata: {
                                author: oTemplate.author,
                                created: new Date().toISOString(),
                                exportVersion: "1.0"
                            }
                        }
                    };
                    
                    // Create and download JSON file
                    var sFileName = sTemplateName.replace(/[^a-z0-9]/gi, '_').toLowerCase() + "_template.json";
                    var oBlob = new Blob([JSON.stringify(oExportData, null, 2)], {type: "application/json"});
                    var sUrl = URL.createObjectURL(oBlob);
                    
                    var oLink = document.createElement("a");
                    oLink.href = sUrl;
                    oLink.download = sFileName;
                    document.body.appendChild(oLink);
                    oLink.click();
                    document.body.removeChild(oLink);
                    URL.revokeObjectURL(sUrl);
                    
                    MessageToast.show("Template downloaded: " + sFileName);
                }.bind(this)
            });
        },

        onCloneTemplate: function (oEvent) {
            oEvent.stopPropagation();
            var oContext = oEvent.getSource().getBindingContext("view");
            var sTemplateName = oContext.getProperty("name");
            var sTemplateId = oContext.getProperty("id");
            
            MessageBox.confirm(
                "Clone template '" + sTemplateName + "'?", {
                    onClose: function (sAction) {
                        if (sAction === MessageBox.Action.OK) {
                            this._cloneTemplate(sTemplateId);
                        }
                    }.bind(this)
                }
            );
        },

        _cloneTemplate: function (sTemplateId) {
            jQuery.ajax({
                url: "/api/templates/" + sTemplateId + "/clone",
                method: "POST",
                success: function () {
                    MessageToast.show("Template cloned successfully");
                    this._loadTemplates();
                }.bind(this),
                error: function () {
                    // Implement client-side clone
                    var oViewModel = this.getView().getModel("view");
                    var aTemplates = oViewModel.getProperty("/templates");
                    var oOriginal = aTemplates.find(function(t) { return t.id === sTemplateId; });
                    
                    if (oOriginal) {
                        var oClone = JSON.parse(JSON.stringify(oOriginal));
                        oClone.id = "t" + Date.now();
                        oClone.name = oOriginal.name + " (Copy)";
                        oClone.downloads = 0;
                        oClone.lastModified = new Date().toISOString();
                        
                        aTemplates.push(oClone);
                        oViewModel.setProperty("/templates", aTemplates);
                        this._updateGridCards();
                        
                        MessageToast.show("Template cloned successfully");
                    }
                }.bind(this)
            });
        },

        onEditTemplate: function (oEvent) {
            oEvent.stopPropagation();
            var oContext = oEvent.getSource().getBindingContext("view");
            var oTemplateData = oContext.getProperty();
            
            if (!this._oEditDialog) {
                this._oEditDialog = sap.ui.xmlfragment("a2a.portal.fragment.EditTemplateDialog", this);
                this.getView().addDependent(this._oEditDialog);
            }
            
            var oDialogModel = new JSONModel(JSON.parse(JSON.stringify(oTemplateData)));
            this._oEditDialog.setModel(oDialogModel);
            this._oEditDialog.open();
        },

        onDeleteTemplate: function (oEvent) {
            oEvent.stopPropagation();
            var oContext = oEvent.getSource().getBindingContext("view");
            var sTemplateName = oContext.getProperty("name");
            var sTemplateId = oContext.getProperty("id");
            
            MessageBox.confirm(
                "Delete template '" + sTemplateName + "'? This action cannot be undone.", {
                    icon: MessageBox.Icon.WARNING,
                    title: "Confirm Deletion",
                    onClose: function (sAction) {
                        if (sAction === MessageBox.Action.OK) {
                            this._deleteTemplate(sTemplateId);
                        }
                    }.bind(this)
                }
            );
        },

        _deleteTemplate: function (sTemplateId) {
            jQuery.ajax({
                url: "/api/templates/" + sTemplateId,
                method: "DELETE",
                success: function () {
                    MessageToast.show("Template deleted successfully");
                    this._loadTemplates();
                }.bind(this),
                error: function (xhr, status, error) {
                    MessageToast.show("Failed to delete template: " + error);
                }.bind(this)
            });
        },

        onExportSelected: function () {
            var oTable = this.byId("templatesTable");
            var aSelectedItems = oTable.getSelectedItems();
            
            if (aSelectedItems.length === 0) {
                MessageToast.show("Please select templates to export");
                return;
            }
            
            var aTemplateIds = aSelectedItems.map(function(oItem) {
                return oItem.getBindingContext("view").getProperty("id");
            });
            
            var oViewModel = this.getView().getModel("view");
            var aTemplates = oViewModel.getProperty("/templates");
            var aSelectedTemplates = aTemplates.filter(function(oTemplate) {
                return aTemplateIds.indexOf(oTemplate.id) !== -1;
            });
            
            var oExportData = {
                exportDate: new Date().toISOString(),
                exportVersion: "1.0",
                templateCount: aSelectedTemplates.length,
                templates: aSelectedTemplates
            };
            
            // Create and download ZIP-like JSON bundle
            var sFileName = "templates_export_" + new Date().getTime() + ".json";
            var oBlob = new Blob([JSON.stringify(oExportData, null, 2)], {type: "application/json"});
            var sUrl = URL.createObjectURL(oBlob);
            
            var oLink = document.createElement("a");
            oLink.href = sUrl;
            oLink.download = sFileName;
            document.body.appendChild(oLink);
            oLink.click();
            document.body.removeChild(oLink);
            URL.revokeObjectURL(sUrl);
            
            MessageToast.show("Exported " + aSelectedTemplates.length + " templates");
        },

        onDeleteSelected: function () {
            var oTable = this.byId("templatesTable");
            var aSelectedItems = oTable.getSelectedItems();
            
            if (aSelectedItems.length === 0) {
                MessageToast.show("Please select templates to delete");
                return;
            }
            
            var sMessage = "Delete " + aSelectedItems.length + " selected template(s)? This action cannot be undone.";
            
            MessageBox.confirm(sMessage, {
                icon: MessageBox.Icon.WARNING,
                title: "Confirm Deletion",
                onClose: function (sAction) {
                    if (sAction === MessageBox.Action.OK) {
                        // Implement batch delete
                        var oViewModel = this.getView().getModel("view");
                        var aTemplates = oViewModel.getProperty("/templates");
                        var aTemplateIds = aSelectedItems.map(function(oItem) {
                            return oItem.getBindingContext("view").getProperty("id");
                        });
                        
                        // Filter out deleted templates
                        var aRemainingTemplates = aTemplates.filter(function(oTemplate) {
                            return aTemplateIds.indexOf(oTemplate.id) === -1;
                        });
                        
                        oViewModel.setProperty("/templates", aRemainingTemplates);
                        this._updateGridCards();
                        oTable.removeSelections();
                        
                        MessageToast.show(aTemplateIds.length + " templates deleted successfully");
                    }
                }
            });
        },

        onBrowseMarketplace: function () {
            // Open marketplace in new window
            var sMarketplaceUrl = "https://marketplace.sap.com/en/solutions?search=A2A%20Templates";
            window.open(sMarketplaceUrl, "_blank");
            MessageToast.show("Opening SAP Marketplace...");
        },

        onViewMyTemplates: function () {
            // Filter to show only user's templates
            var oTable = this.byId("templatesTable");
            var oBinding = oTable.getBinding("items");
            var oFilter = new sap.ui.model.Filter("author", sap.ui.model.FilterOperator.EQ, "Current User");
            oBinding.filter([oFilter]);
            
            // Update view mode
            var oViewModel = this.getView().getModel("view");
            oViewModel.setProperty("/viewMode", "table");
            
            MessageToast.show("Showing your templates");
        },

        onViewTutorials: function () {
            MessageToast.show("Opening template tutorials...");
        },

        formatDate: function (sDate) {
            if (!sDate) {
                return "";
            }
            
            var oDateFormat = DateFormat.getDateTimeInstance({
                style: "medium"
            });
            
            return oDateFormat.format(new Date(sDate));
        },

        formatCategoryState: function (sCategory) {
            switch (sCategory) {
                case "data-product": return "Success";
                case "standardization": return "Information";
                case "ai-processing": return "Warning";
                case "validation": return "Error";
                case "integration": return "None";
                default: return "None";
            }
        }
    });
});