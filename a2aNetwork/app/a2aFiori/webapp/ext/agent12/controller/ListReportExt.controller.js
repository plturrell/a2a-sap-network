sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Fragment"
], function(ControllerExtension, MessageToast, MessageBox, Fragment) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent12.ext.controller.ListReportExt", {
        
        // Catalog Dashboard Action
        onCatalogDashboard: function() {
            if (!this._catalogDashboard) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent12.ext.fragment.CatalogDashboard",
                    controller: this
                }).then(function(oDialog) {
                    this._catalogDashboard = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadDashboardData();
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadDashboardData();
                this._catalogDashboard.open();
            }
        },

        // Create New Catalog Entry
        onCreateCatalogEntry: function() {
            if (!this._createEntryDialog) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent12.ext.fragment.CreateCatalogEntry",
                    controller: this
                }).then(function(oDialog) {
                    this._createEntryDialog = oDialog;
                    this.getView().addDependent(oDialog);
                    oDialog.open();
                }.bind(this));
            } else {
                this._createEntryDialog.open();
            }
        },

        // Service Discovery Action
        onServiceDiscovery: function() {
            if (!this._serviceDiscovery) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent12.ext.fragment.ServiceDiscovery",
                    controller: this
                }).then(function(oDialog) {
                    this._serviceDiscovery = oDialog;
                    this.getView().addDependent(oDialog);
                    this._initializeDiscovery();
                    oDialog.open();
                }.bind(this));
            } else {
                this._initializeDiscovery();
                this._serviceDiscovery.open();
            }
        },

        // Registry Manager Action
        onRegistryManager: function() {
            if (!this._registryManager) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent12.ext.fragment.RegistryManager",
                    controller: this
                }).then(function(oDialog) {
                    this._registryManager = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadRegistryData();
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadRegistryData();
                this._registryManager.open();
            }
        },

        // Category Manager Action
        onCategoryManager: function() {
            if (!this._categoryManager) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent12.ext.fragment.CategoryManager",
                    controller: this
                }).then(function(oDialog) {
                    this._categoryManager = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadCategoryData();
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadCategoryData();
                this._categoryManager.open();
            }
        },

        // Search Manager Action
        onSearchManager: function() {
            if (!this._searchManager) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent12.ext.fragment.SearchManager",
                    controller: this
                }).then(function(oDialog) {
                    this._searchManager = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadSearchIndexData();
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadSearchIndexData();
                this._searchManager.open();
            }
        },

        // Metadata Editor Action
        onMetadataEditor: function() {
            const oBinding = this.base.getView().byId("fe::table::CatalogEntries::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectEntriesFirst"));
                return;
            }

            if (!this._metadataEditor) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent12.ext.fragment.MetadataEditor",
                    controller: this
                }).then(function(oDialog) {
                    this._metadataEditor = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadMetadataForEditing(aSelectedContexts[0]);
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadMetadataForEditing(aSelectedContexts[0]);
                this._metadataEditor.open();
            }
        },

        // Discovery Scanner Action
        onDiscoveryScanner: function() {
            MessageBox.confirm(
                this.getResourceBundle().getText("msg.startDiscoveryScan"),
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._startDiscoveryScan();
                        }
                    }.bind(this)
                }
            );
        },

        // Bulk Validation Action
        onBulkValidation: function() {
            const oBinding = this.base.getView().byId("fe::table::CatalogEntries::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectEntriesFirst"));
                return;
            }

            this._validateBulkEntries(aSelectedContexts);
        },

        // Bulk Publishing Action
        onBulkPublishing: function() {
            const oBinding = this.base.getView().byId("fe::table::CatalogEntries::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectEntriesFirst"));
                return;
            }

            MessageBox.confirm(
                this.getResourceBundle().getText("msg.publishEntriesConfirm", [aSelectedContexts.length]),
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._publishBulkEntries(aSelectedContexts);
                        }
                    }.bind(this)
                }
            );
        },

        // Real-time Updates via WebSocket
        onAfterRendering: function() {
            this._initializeWebSocket();
        },

        _initializeWebSocket: function() {
            if (this._ws) return;

            try {
                this._ws = new WebSocket('ws://localhost:8012/catalog/updates');
                
                this._ws.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    this._handleCatalogUpdate(data);
                }.bind(this);

                this._ws.onclose = function() {
                    setTimeout(() => this._initializeWebSocket(), 5000);
                }.bind(this);

            } catch (error) {
                console.warn("WebSocket connection failed, falling back to polling");
                this._initializePolling();
            }
        },

        _initializePolling: function() {
            this._pollInterval = setInterval(() => {
                this._refreshCatalogData();
            }, 10000);
        },

        _handleCatalogUpdate: function(data) {
            const oModel = this.getView().getModel();
            
            switch (data.type) {
                case 'RESOURCE_REGISTERED':
                    MessageToast.show(this.getResourceBundle().getText("msg.resourceRegistered"));
                    this._refreshCatalogData();
                    break;
                case 'DISCOVERY_COMPLETED':
                    MessageToast.show(this.getResourceBundle().getText("msg.discoveryCompleted"));
                    this._refreshCatalogData();
                    break;
                case 'INDEX_UPDATED':
                    MessageToast.show(this.getResourceBundle().getText("msg.searchIndexUpdated"));
                    break;
                case 'REGISTRY_SYNCED':
                    MessageToast.show(this.getResourceBundle().getText("msg.registrySynced"));
                    this._refreshCatalogData();
                    break;
                case 'METADATA_UPDATED':
                    this._refreshCatalogData();
                    break;
            }
        },

        _loadDashboardData: function() {
            const oModel = this.getView().getModel();
            
            // Load catalog statistics
            oModel.callFunction("/GetCatalogStatistics", {
                success: function(data) {
                    this._updateDashboardCharts(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingStatistics"));
                }.bind(this)
            });
        },

        _initializeDiscovery: function() {
            const oModel = this.getView().getModel();
            
            oModel.callFunction("/GetDiscoveryMethods", {
                success: function(data) {
                    this._updateDiscoveryOptions(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingDiscoveryMethods"));
                }.bind(this)
            });
        },

        _loadRegistryData: function() {
            const oModel = this.getView().getModel();
            
            oModel.callFunction("/GetRegistryConfigurations", {
                success: function(data) {
                    this._updateRegistryList(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingRegistryData"));
                }.bind(this)
            });
        },

        _loadCategoryData: function() {
            const oModel = this.getView().getModel();
            
            oModel.callFunction("/GetServiceCategories", {
                success: function(data) {
                    this._updateCategoryTree(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingCategoryData"));
                }.bind(this)
            });
        },

        _loadSearchIndexData: function() {
            const oModel = this.getView().getModel();
            
            oModel.callFunction("/GetSearchIndexes", {
                success: function(data) {
                    this._updateSearchIndexList(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingSearchIndexData"));
                }.bind(this)
            });
        },

        _loadMetadataForEditing: function(oContext) {
            const oModel = this.getView().getModel();
            const sEntryId = oContext.getObject().entryId;
            
            oModel.callFunction("/GetMetadataProperties", {
                urlParameters: {
                    entryId: sEntryId
                },
                success: function(data) {
                    this._displayMetadataEditor(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingMetadata"));
                }.bind(this)
            });
        },

        _startDiscoveryScan: function() {
            const oModel = this.getView().getModel();
            
            MessageToast.show(this.getResourceBundle().getText("msg.discoveryScanStarted"));
            
            oModel.callFunction("/StartResourceDiscovery", {
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.discoveryCompleted"));
                    this._refreshCatalogData();
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.discoveryFailed"));
                }.bind(this)
            });
        },

        _validateBulkEntries: function(aSelectedContexts) {
            const oModel = this.getView().getModel();
            const aEntryIds = aSelectedContexts.map(ctx => ctx.getObject().entryId);
            
            MessageToast.show(this.getResourceBundle().getText("msg.bulkValidationStarted", [aEntryIds.length]));
            
            oModel.callFunction("/ValidateCatalogEntries", {
                urlParameters: {
                    entryIds: aEntryIds.join(',')
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.bulkValidationCompleted"));
                    this._refreshCatalogData();
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.validationFailed"));
                }.bind(this)
            });
        },

        _publishBulkEntries: function(aSelectedContexts) {
            const oModel = this.getView().getModel();
            const aEntryIds = aSelectedContexts.map(ctx => ctx.getObject().entryId);
            
            MessageToast.show(this.getResourceBundle().getText("msg.bulkPublishingStarted", [aEntryIds.length]));
            
            oModel.callFunction("/PublishCatalogEntries", {
                urlParameters: {
                    entryIds: aEntryIds.join(',')
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.bulkPublishingCompleted"));
                    this._refreshCatalogData();
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.publishingFailed"));
                }.bind(this)
            });
        },

        _refreshCatalogData: function() {
            const oBinding = this.base.getView().byId("fe::table::CatalogEntries::LineItem").getBinding("rows");
            oBinding.refresh();
        },

        _updateDashboardCharts: function(data) {
            // Update catalog growth chart
            // Update resource type distribution chart
            // Update category distribution chart
        },

        _updateDiscoveryOptions: function(data) {
            // Update discovery method options
        },

        _updateRegistryList: function(data) {
            // Update registry configuration list
        },

        _updateCategoryTree: function(data) {
            // Update category tree structure
        },

        _updateSearchIndexList: function(data) {
            // Update search index list
        },

        _displayMetadataEditor: function(data) {
            // Display metadata properties for editing
        },

        getResourceBundle: function() {
            return this.getView().getModel("i18n").getResourceBundle();
        },

        onExit: function() {
            if (this._ws) {
                this._ws.close();
            }
            if (this._pollInterval) {
                clearInterval(this._pollInterval);
            }
        }
    });
});