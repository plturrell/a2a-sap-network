sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Fragment",
    "a2a/network/agent12/ext/utils/SecurityUtils"
], function(ControllerExtension, MessageToast, MessageBox, Fragment, SecurityUtils) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent12.ext.controller.ObjectPageExt", {
        
        // Register Resource Action
        onRegisterResource: function() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            
            if (oData.status === 'active') {
                MessageToast.show(this.getResourceBundle().getText("msg.resourceAlreadyRegistered"));
                return;
            }

            MessageBox.confirm(
                this.getResourceBundle().getText("msg.registerResourceConfirm"),
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._registerResource(oContext);
                        }
                    }.bind(this)
                }
            );
        },

        // Update Metadata Action
        onUpdateMetadata: function() {
            const oContext = this.base.getView().getBindingContext();
            
            if (!this._metadataEditor) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent12.ext.fragment.MetadataEditor",
                    controller: this
                }).then(function(oDialog) {
                    this._metadataEditor = oDialog;
                    this.getView().addDependent(oDialog);
                    this._loadMetadataForEditing(oContext);
                    oDialog.open();
                }.bind(this));
            } else {
                this._loadMetadataForEditing(oContext);
                this._metadataEditor.open();
            }
        },

        // Validate Entry Action
        onValidateEntry: function() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            
            if (!oData.resourceUrl || oData.resourceUrl.trim() === '') {
                MessageToast.show(this.getResourceBundle().getText("error.noResourceUrl"));
                return;
            }

            this._validateEntry(oContext);
        },

        // Publish Entry Action
        onPublishEntry: function() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            
            if (oData.validationStatus !== 'validated') {
                MessageToast.show(this.getResourceBundle().getText("error.entryNotValidated"));
                return;
            }

            MessageBox.confirm(
                this.getResourceBundle().getText("msg.publishEntryConfirm"),
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._publishEntry(oContext);
                        }
                    }.bind(this)
                }
            );
        },

        // Index Resource Action
        onIndexResource: function() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            
            if (!oData.searchable) {
                MessageToast.show(this.getResourceBundle().getText("error.resourceNotSearchable"));
                return;
            }

            this._indexResource(oContext);
        },

        // Discover Dependencies Action
        onDiscoverDependencies: function() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            
            if (!oData.resourceUrl || oData.resourceUrl.trim() === '') {
                MessageToast.show(this.getResourceBundle().getText("error.noResourceUrl"));
                return;
            }

            if (!this._dependencyDiscovery) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent12.ext.fragment.DependencyDiscovery",
                    controller: this
                }).then(function(oDialog) {
                    this._dependencyDiscovery = oDialog;
                    this.getView().addDependent(oDialog);
                    this._startDependencyDiscovery(oContext);
                    oDialog.open();
                }.bind(this));
            } else {
                this._startDependencyDiscovery(oContext);
                this._dependencyDiscovery.open();
            }
        },

        // Export Catalog Action
        onExportCatalog: function() {
            const oContext = this.base.getView().getBindingContext();
            
            if (!this._catalogExporter) {
                Fragment.load({
                    id: this.getView().getId(),
                    name: "a2a.network.agent12.ext.fragment.CatalogExporter",
                    controller: this
                }).then(function(oDialog) {
                    this._catalogExporter = oDialog;
                    this.getView().addDependent(oDialog);
                    oDialog.open();
                }.bind(this));
            } else {
                this._catalogExporter.open();
            }
        },

        // Sync Registry Action
        onSyncRegistry: function() {
            const oContext = this.base.getView().getBindingContext();
            const oData = oContext.getObject();
            
            if (oData.registrationSource === 'manual') {
                MessageToast.show(this.getResourceBundle().getText("error.manualEntryCannotSync"));
                return;
            }

            MessageBox.confirm(
                this.getResourceBundle().getText("msg.syncRegistryConfirm"),
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._syncRegistry(oContext);
                        }
                    }.bind(this)
                }
            );
        },

        // Real-time monitoring initialization
        onAfterRendering: function() {
            this._initializeEntryMonitoring();
        },

        _initializeEntryMonitoring: function() {
            const oContext = this.base.getView().getBindingContext();
            if (!oContext) return;

            const entryId = oContext.getObject().entryId;
            
            // Subscribe to catalog entry updates for this specific entry
            if (this._eventSource) {
                this._eventSource.close();
            }

            try {
                this._eventSource = SecurityUtils.createSecureEventSource(`https://localhost:8012/catalog/${entryId}/stream`, {});
                
                if (this._eventSource) {
                    this._eventSource.addEventListener('validation-progress', (event) => {
                        const data = JSON.parse(event.data);
                        this._updateValidationProgress(data);
                    });
                }

                if (this._eventSource) {
                    this._eventSource.addEventListener('discovery-progress', (event) => {
                        const data = JSON.parse(event.data);
                        this._updateDiscoveryProgress(data);
                    });
                }

                if (this._eventSource) {
                    this._eventSource.addEventListener('metadata-updated', (event) => {
                        const data = JSON.parse(event.data);
                        this._handleMetadataUpdate(data);
                    });
                }

                if (this._eventSource) {
                    this._eventSource.addEventListener('status-changed', (event) => {
                        const data = JSON.parse(event.data);
                        this._handleStatusChange(data);
                    });
                }

            } catch (error) {
                console.warn("Server-Sent Events not available, using polling");
                this._initializePolling(entryId);
            }
        },

        _initializePolling: function(entryId) {
            this._pollInterval = setInterval(() => {
                this._refreshEntryData();
            }, 5000);
        },

        _registerResource: function(oContext) {
            const oModel = this.getView().getModel();
            const sEntryId = oContext.getObject().entryId;
            
            MessageToast.show(this.getResourceBundle().getText("msg.registrationStarted"));
            
            if (!SecurityUtils.checkCatalogAuth('RegisterResource', {})) {
                return;
            }

            SecurityUtils.secureCallFunction(oModel, "/RegisterResource", {
                urlParameters: {
                    entryId: sEntryId
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.resourceRegistered"));
                    this._refreshEntryData();
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.registrationFailed"));
                }.bind(this)
            });
        },

        _validateEntry: function(oContext) {
            const oModel = this.getView().getModel();
            const sEntryId = oContext.getObject().entryId;
            
            MessageToast.show(this.getResourceBundle().getText("msg.validationStarted"));
            
            if (!SecurityUtils.checkCatalogAuth('ValidateEntry', {})) {
                return;
            }

            SecurityUtils.secureCallFunction(oModel, "/ValidateEntry", {
                urlParameters: {
                    entryId: sEntryId
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.entryValidated"));
                    this._refreshEntryData();
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.validationFailed"));
                }.bind(this)
            });
        },

        _publishEntry: function(oContext) {
            const oModel = this.getView().getModel();
            const sEntryId = oContext.getObject().entryId;
            
            MessageToast.show(this.getResourceBundle().getText("msg.publishingStarted"));
            
            if (!SecurityUtils.checkCatalogAuth('PublishEntry', {})) {
                return;
            }

            SecurityUtils.secureCallFunction(oModel, "/PublishEntry", {
                urlParameters: {
                    entryId: sEntryId
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.entryPublished"));
                    this._refreshEntryData();
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.publishingFailed"));
                }.bind(this)
            });
        },

        _indexResource: function(oContext) {
            const oModel = this.getView().getModel();
            const sEntryId = oContext.getObject().entryId;
            
            MessageToast.show(this.getResourceBundle().getText("msg.indexingStarted"));
            
            if (!SecurityUtils.checkCatalogAuth('IndexResource', {})) {
                return;
            }

            SecurityUtils.secureCallFunction(oModel, "/IndexResource", {
                urlParameters: {
                    entryId: sEntryId
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.resourceIndexed"));
                    this._refreshEntryData();
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.indexingFailed"));
                }.bind(this)
            });
        },

        _startDependencyDiscovery: function(oContext) {
            const oModel = this.getView().getModel();
            const sEntryId = oContext.getObject().entryId;
            
            MessageToast.show(this.getResourceBundle().getText("msg.dependencyDiscoveryStarted"));
            
            if (!SecurityUtils.checkCatalogAuth('DiscoverDependencies', {})) {
                return;
            }

            SecurityUtils.secureCallFunction(oModel, "/DiscoverDependencies", {
                urlParameters: {
                    entryId: sEntryId
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.dependenciesDiscovered"));
                    this._refreshEntryData();
                    this._displayDependencyResults(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.dependencyDiscoveryFailed"));
                }.bind(this)
            });
        },

        _syncRegistry: function(oContext) {
            const oModel = this.getView().getModel();
            const sEntryId = oContext.getObject().entryId;
            
            MessageToast.show(this.getResourceBundle().getText("msg.registrySyncStarted"));
            
            if (!SecurityUtils.checkCatalogAuth('SyncRegistryEntry', {})) {
                return;
            }

            SecurityUtils.secureCallFunction(oModel, "/SyncRegistryEntry", {
                urlParameters: {
                    entryId: sEntryId
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.registrySynced"));
                    this._refreshEntryData();
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.syncFailed"));
                }.bind(this)
            });
        },

        _loadMetadataForEditing: function(oContext) {
            const oModel = this.getView().getModel();
            const sEntryId = oContext.getObject().entryId;
            
            SecurityUtils.secureCallFunction(oModel, "/GetMetadataProperties", {
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

        _updateValidationProgress: function(data) {
            // Update validation progress indicators
            const oProgressIndicator = this.getView().byId("validationProgress");
            if (oProgressIndicator) {
                oProgressIndicator.setPercentValue(data.progress);
                oProgressIndicator.setDisplayValue(`${data.progress}% - ${data.currentStep}`);
            }
        },

        _updateDiscoveryProgress: function(data) {
            // Update discovery progress indicators
            const oProgressIndicator = this.getView().byId("discoveryProgress");
            if (oProgressIndicator) {
                oProgressIndicator.setPercentValue(data.progress);
                oProgressIndicator.setDisplayValue(`${data.progress}% - ${data.currentStep}`);
            }
        },

        _handleMetadataUpdate: function(data) {
            // Sanitize metadata update data
            const sanitizedData = SecurityUtils.sanitizeCatalogData(JSON.stringify(data));
            MessageToast.show(this.getResourceBundle().getText("msg.metadataUpdated"));
            this._refreshEntryData();
        },

        _handleStatusChange: function(data) {
            MessageToast.show(this.getResourceBundle().getText("msg.statusChanged", [data.newStatus]));
            this._refreshEntryData();
        },

        _refreshEntryData: function() {
            const oContext = this.base.getView().getBindingContext();
            if (oContext) {
                oContext.refresh();
            }
        },

        _displayMetadataEditor: function(data) {
            // Display metadata properties for editing
        },

        _displayDependencyResults: function(data) {
            // Display discovered dependencies
        },

        getResourceBundle: function() {
            return this.getView().getModel("i18n").getResourceBundle();
        },

        onExit: function() {
            if (this._eventSource) {
                this._eventSource.close();
            }
            if (this._pollInterval) {
                clearInterval(this._pollInterval);
            }
        }
    });
});