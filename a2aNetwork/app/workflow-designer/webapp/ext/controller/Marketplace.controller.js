sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/Card",
    "sap/f/cards/NumericHeader",
    "sap/f/cards/NumericSideIndicator",
    "sap/m/VBox",
    "sap/m/HBox",
    "sap/m/Text",
    "sap/m/Label",
    "sap/m/RatingIndicator",
    "sap/m/Button",
    "sap/m/MessageToast",
    "sap/ui/core/Fragment"
], function (Controller, JSONModel, Card, NumericHeader, NumericSideIndicator, VBox, HBox, Text, Label, RatingIndicator, Button, MessageToast, Fragment) {
    "use strict";

    return Controller.extend("a2a.workflow.designer.ext.controller.Marketplace", {

        onInit: function () {
            this._setupModel();
            this._loadMarketplaceData();
        },

        _setupModel: function () {
            const oModel = new JSONModel({
                templates: [],
                filteredTemplates: [],
                filters: {
                    search: "",
                    category: "all",
                    minRating: 0,
                    sortBy: "popularity"
                }
            });
            this.getView().setModel(oModel, "marketplace");
        },

        _loadMarketplaceData: function () {
            const oModel = this.getView().getModel();
            const that = this;

            // Call the search function to get marketplace data
            oModel.callFunction("/searchTemplates", {
                urlParameters: {
                    query: "",
                    category: "",
                    minRating: 0
                },
                success: function (oData) {
                    const templates = oData?.value || oData?.results || [];
                    that._processMarketplaceData(templates);
                },
                error: function (oError) {
                    console.error("Error loading marketplace data:", oError);
                    MessageToast.show("Unable to connect to backend. Loading sample data.");
                    // Load mock data as fallback
                    that._loadMockData();
                }
            });
        },

        _loadMockData: function () {
            const mockTemplates = [
                {
                    ID: "1",
                    name: "Document Grounding Pipeline",
                    description: "Complete document processing pipeline with chunking, embedding, and indexing",
                    categoryName: "Document Processing",
                    categoryIcon: "sap-icon://document",
                    rating: 4.8,
                    ratingCount: 125,
                    usageCount: 1250,
                    ratingBadge: "Excellent",
                    popularityBadge: "Popular",
                    author: "A2A Team",
                    estimatedDuration: 5,
                    requiredAgents: [0, 1, 2, 3],
                    tags: ["document", "grounding", "embeddings", "RAG"],
                    isOfficial: true,
                    version: "1.0.0"
                },
                {
                    ID: "2",
                    name: "Simple RAG Workflow",
                    description: "Basic retrieval-augmented generation workflow",
                    categoryName: "RAG Pipelines",
                    categoryIcon: "sap-icon://process",
                    rating: 4.5,
                    ratingCount: 89,
                    usageCount: 890,
                    ratingBadge: "Excellent",
                    popularityBadge: "Trending",
                    author: "Community",
                    estimatedDuration: 2,
                    requiredAgents: [2, 3, 9],
                    tags: ["RAG", "retrieval", "generation", "search"],
                    isOfficial: false,
                    version: "1.2.1"
                },
                {
                    ID: "3",
                    name: "Data Quality Pipeline",
                    description: "Comprehensive data quality assessment and improvement workflow",
                    categoryName: "Quality Assurance",
                    categoryIcon: "sap-icon://quality-issue",
                    rating: 4.2,
                    ratingCount: 67,
                    usageCount: 423,
                    ratingBadge: "Good",
                    popularityBadge: "Trending",
                    author: "QA Team",
                    estimatedDuration: 8,
                    requiredAgents: [0, 1, 5, 6],
                    tags: ["quality", "validation", "assessment", "improvement"],
                    isOfficial: true,
                    version: "2.1.0"
                }
            ];

            const oMarketplaceModel = this.getView().getModel("marketplace");
            oMarketplaceModel.setProperty("/templates", mockTemplates);
            oMarketplaceModel.setProperty("/filteredTemplates", mockTemplates);
            this._renderTemplateCards(mockTemplates);
        },

        _processMarketplaceData: function (aTemplates) {
            const oMarketplaceModel = this.getView().getModel("marketplace");
            oMarketplaceModel.setProperty("/templates", aTemplates);
            oMarketplaceModel.setProperty("/filteredTemplates", aTemplates);
            this._renderTemplateCards(aTemplates);
        },

        _renderTemplateCards: function (aTemplates) {
            const oGrid = this.byId("templateGrid");
            oGrid.destroyItems();

            aTemplates.forEach(function (oTemplate) {
                const oCard = this._createTemplateCard(oTemplate);
                oGrid.addItem(oCard);
            }.bind(this));
        },

        _createTemplateCard: function (oTemplate) {
            const oCard = new Card({
                class: "sapUiMediumMargin",
                header: new NumericHeader({
                    title: oTemplate.name,
                    subtitle: oTemplate.categoryName,
                    number: oTemplate.rating.toString(),
                    scale: "★",
                    state: this._getRatingState(oTemplate.rating),
                    details: oTemplate.ratingCount + " reviews",
                    sideIndicators: [
                        new NumericSideIndicator({
                            title: "Usage",
                            number: oTemplate.usageCount.toString(),
                            unit: "times"
                        })
                    ]
                }),
                content: new VBox({
                    class: "sapUiMediumMargin",
                    items: [
                        new Text({
                            text: oTemplate.description,
                            class: "sapUiSmallMarginBottom"
                        }),
                        new HBox({
                            justifyContent: "SpaceBetween",
                            alignItems: "Center",
                            class: "sapUiSmallMarginBottom",
                            items: [
                                new Label({
                                    text: "Duration: " + oTemplate.estimatedDuration + " min"
                                }),
                                new Text({
                                    text: oTemplate.popularityBadge,
                                    class: this._getPopularityBadgeClass(oTemplate.popularityBadge)
                                })
                            ]
                        }),
                        new HBox({
                            class: "sapUiSmallMarginBottom",
                            items: [
                                new Label({
                                    text: "Required Agents: "
                                }),
                                new Text({
                                    text: oTemplate.requiredAgents.join(", ")
                                })
                            ]
                        }),
                        new HBox({
                            justifyContent: "SpaceBetween",
                            alignItems: "Center",
                            items: [
                                new Text({
                                    text: "by " + oTemplate.author + " • v" + oTemplate.version
                                }),
                                new HBox({
                                    items: [
                                        new Button({
                                            text: "Preview",
                                            type: "Transparent",
                                            icon: "sap-icon://show",
                                            press: [this.onPreviewTemplate, this]
                                        }).data("templateId", oTemplate.ID),
                                        new Button({
                                            text: "Use Template",
                                            type: "Emphasized",
                                            icon: "sap-icon://download",
                                            press: [this.onUseTemplate, this]
                                        }).data("templateId", oTemplate.ID)
                                    ]
                                })
                            ]
                        })
                    ]
                })
            });

            return oCard;
        },

        _getRatingState: function (fRating) {
            if (fRating >= 4.5) return "Good";
            if (fRating >= 3.5) return "Neutral";
            return "Error";
        },

        _getPopularityBadgeClass: function (sBadge) {
            switch (sBadge) {
                case "Popular": return "sapUiPositiveText";
                case "Trending": return "sapUiCriticalText";
                case "New": return "sapUiInformativeText";
                default: return "";
            }
        },

        onSearch: function (oEvent) {
            this._applyFilters();
        },

        onSearchLiveChange: function (oEvent) {
            const oMarketplaceModel = this.getView().getModel("marketplace");
            oMarketplaceModel.setProperty("/filters/search", oEvent.getParameter("newValue"));
            this._applyFilters();
        },

        onCategoryChange: function (oEvent) {
            const oMarketplaceModel = this.getView().getModel("marketplace");
            const sSelectedKey = oEvent.getParameter("selectedItem").getKey();
            oMarketplaceModel.setProperty("/filters/category", sSelectedKey);
            this._applyFilters();
        },

        onRatingChange: function (oEvent) {
            const oMarketplaceModel = this.getView().getModel("marketplace");
            oMarketplaceModel.setProperty("/filters/minRating", oEvent.getParameter("value"));
            this._applyFilters();
        },

        _applyFilters: function () {
            const oMarketplaceModel = this.getView().getModel("marketplace");
            const oFilters = oMarketplaceModel.getProperty("/filters");
            const aTemplates = oMarketplaceModel.getProperty("/templates");

            const aFilteredTemplates = aTemplates.filter(function (oTemplate) {
                // Search filter
                if (oFilters.search && !oTemplate.name.toLowerCase().includes(oFilters.search.toLowerCase()) &&
                    !oTemplate.description.toLowerCase().includes(oFilters.search.toLowerCase())) {
                    return false;
                }

                // Category filter - handle both category_ID and categoryName
                const categoryMatch = oFilters.category === "all" || 
                    oTemplate.category_ID === oFilters.category ||
                    oTemplate.categoryName === oFilters.category;
                if (!categoryMatch) {
                    return false;
                }

                // Rating filter
                if (oTemplate.rating < oFilters.minRating) {
                    return false;
                }

                return true;
            });

            // Apply sorting
            this._sortTemplates(aFilteredTemplates, oFilters.sortBy);

            oMarketplaceModel.setProperty("/filteredTemplates", aFilteredTemplates);
            this._renderTemplateCards(aFilteredTemplates);
        },

        _sortTemplates: function (aTemplates, sSortBy) {
            aTemplates.sort(function (a, b) {
                switch (sSortBy) {
                    case "rating":
                        return b.rating - a.rating;
                    case "newest":
                        return new Date(b.createdAt || Date.now()) - new Date(a.createdAt || Date.now());
                    case "usage":
                        return b.usageCount - a.usageCount;
                    case "popularity":
                    default:
                        return b.usageCount - a.usageCount; // Default to usage count
                }
            });
        },

        onCreateTemplate: function () {
            const oRouter = this.getOwnerComponent().getRouter();
            oRouter.navTo("TemplateDetail", {
                templateId: "new"
            });
        },

        onImportTemplate: function () {
            MessageToast.show("Import functionality will be available soon");
            // TODO: Implement template import dialog
        },

        onPreviewTemplate: function (oEvent) {
            const sTemplateId = oEvent.getSource().data("templateId");
            
            if (!this._oPreviewDialog) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.workflow.designer.ext.fragment.TemplatePreview",
                    controller: this
                }).then(function (oDialog) {
                    this._oPreviewDialog = oDialog;
                    this.getView().addDependent(oDialog);
                    this._showPreview(sTemplateId);
                }.bind(this));
            } else {
                this._showPreview(sTemplateId);
            }
        },

        _showPreview: function (sTemplateId) {
            const oMarketplaceModel = this.getView().getModel("marketplace");
            const aTemplates = oMarketplaceModel.getProperty("/templates");
            const oTemplate = aTemplates.find(function (t) { return t.ID === sTemplateId; });

            if (oTemplate) {
                const oPreviewModel = new JSONModel(oTemplate);
                this._oPreviewDialog.setModel(oPreviewModel, "preview");
                this._oPreviewDialog.open();
            }
        },

        onClosePreview: function () {
            this._oPreviewDialog.close();
        },

        onUseTemplate: function (oEvent) {
            const sTemplateId = oEvent.getSource().data("templateId");
            
            if (!this._oCreateWorkflowDialog) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.workflow.designer.ext.fragment.CreateWorkflowDialog",
                    controller: this
                }).then(function (oDialog) {
                    this._oCreateWorkflowDialog = oDialog;
                    this.getView().addDependent(oDialog);
                    this._showCreateWorkflowDialog(sTemplateId);
                }.bind(this));
            } else {
                this._showCreateWorkflowDialog(sTemplateId);
            }
        },

        _showCreateWorkflowDialog: function (sTemplateId) {
            const oMarketplaceModel = this.getView().getModel("marketplace");
            const aTemplates = oMarketplaceModel.getProperty("/templates");
            const oTemplate = aTemplates.find(function (t) { return t.ID === sTemplateId; });

            if (oTemplate) {
                const oDialogModel = new JSONModel({
                    template: oTemplate,
                    workflow: {
                        name: oTemplate.name + " Instance",
                        description: "Created from " + oTemplate.name + " template",
                        parameters: {}
                    }
                });
                this._oCreateWorkflowDialog.setModel(oDialogModel, "dialog");
                this._oCreateWorkflowDialog.open();
            }
        },

        onConfirmCreateWorkflow: function () {
            const oDialogModel = this._oCreateWorkflowDialog.getModel("dialog");
            const oData = oDialogModel.getData();
            const oModel = this.getView().getModel();

            oModel.callFunction("/createWorkflowFromTemplate", {
                urlParameters: {
                    templateId: oData.template.ID,
                    name: oData.workflow.name,
                    description: oData.workflow.description,
                    parameters: JSON.stringify(oData.workflow.parameters)
                },
                success: function (oResponse) {
                    MessageToast.show("Workflow created successfully!");
                    this._oCreateWorkflowDialog.close();
                    
                    // Navigate to the new workflow instance
                    const oRouter = this.getOwnerComponent().getRouter();
                    const instanceId = oResponse?.createWorkflowFromTemplate?.ID || oResponse?.ID || 'new';
                    oRouter.navTo("WorkflowInstanceDetail", {
                        instanceId: instanceId
                    });
                }.bind(this),
                error: function (oError) {
                    console.error("Error creating workflow:", oError);
                    const errorMessage = oError?.message || oError?.responseText || "Unknown error occurred";
                    MessageToast.show("Error creating workflow: " + errorMessage);
                }
            });
        },

        onCancelCreateWorkflow: function () {
            this._oCreateWorkflowDialog.close();
        },

        onSortChange: function (oEvent) {
            const sSelectedKey = oEvent.getParameter("item").getKey();
            const oMarketplaceModel = this.getView().getModel("marketplace");
            oMarketplaceModel.setProperty("/filters/sortBy", sSelectedKey);
            this._applyFilters();
        }

    });
});