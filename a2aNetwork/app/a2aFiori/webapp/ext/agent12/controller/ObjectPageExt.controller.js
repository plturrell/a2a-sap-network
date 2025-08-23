sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel",
    "a2a/network/agent12/ext/utils/SecurityUtils"
], function(ControllerExtension, MessageToast, MessageBox, Fragment, JSONModel, SecurityUtils) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent12.ext.controller.ObjectPageExt", {

        override: {
            onInit: function () {
                this._initializeCreateModel();
            }
        },

        _initializeCreateModel: function() {
            var oCreateData = {
                resourceName: "",
                description: "",
                catalogType: "",
                resourceType: "",
                category: "",
                visibility: "internal",
                resourceNameState: "",
                resourceNameStateText: "",
                catalogTypeState: "",
                catalogTypeStateText: "",
                resourceTypeState: "",
                resourceTypeStateText: "",
                categoryState: "",
                categoryStateText: "",
                resourceUrl: "",
                resourceUrlState: "",
                resourceUrlStateText: "",
                version: "",
                apiVersion: "",
                protocol: "https",
                authenticationMethod: "bearer",
                contentType: "json",
                documentation: "",
                healthCheckUrl: "",
                swaggerUrl: "",
                metadataSchema: "dublincore",
                keywords: "",
                tags: "",
                owner: "",
                maintainer: "",
                license: "",
                compliance: [],
                dataClassification: "internal",
                retentionPeriod: 365,
                searchable: true,
                searchTags: "",
                searchWeight: 50,
                autoDiscovery: false,
                discoveryFrequency: "daily",
                categories: [],
                metadataProperties: [],
                encryptionRequired: false,
                auditingEnabled: false,
                complianceScore: 0,
                sensitiveDataDetected: false
            };
            var oCreateModel = new JSONModel(oCreateData);
            this.getView().setModel(oCreateModel, "create");
        },

        onCreateCatalogEntry: function() {
            var oView = this.getView();
            
            if (!this._oCreateDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent12.ext.fragment.CreateCatalogEntry",
                    controller: this
                }).then(function(oDialog) {
                    this._oCreateDialog = oDialog;
                    oView.addDependent(this._oCreateDialog);
                    this._loadCategoriesForCreate();
                    this._oCreateDialog.open();
                }.bind(this));
            } else {
                this._loadCategoriesForCreate();
                this._oCreateDialog.open();
            }
        },

        _loadCategoriesForCreate: function() {
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            // Load categories (would be from service in real implementation)
            oData.categories = [
                { categoryId: "api", categoryName: "APIs" },
                { categoryId: "service", categoryName: "Services" },
                { categoryId: "data", categoryName: "Data Sources" },
                { categoryId: "infrastructure", categoryName: "Infrastructure" },
                { categoryId: "security", categoryName: "Security Services" }
            ];
            
            oCreateModel.setData(oData);
        },

        onResourceNameChange: function(oEvent) {
            var sValue = oEvent.getParameter("value");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            // Sanitize input
            var sSanitized = SecurityUtils.sanitizeCatalogData(sValue);
            if (sSanitized !== sValue) {
                oEvent.getSource().setValue(sSanitized);
                sValue = sSanitized;
            }
            
            if (!sValue || sValue.length < 3) {
                oData.resourceNameState = "Error";
                oData.resourceNameStateText = "Resource name must be at least 3 characters";
            } else if (sValue.length > 100) {
                oData.resourceNameState = "Error";
                oData.resourceNameStateText = "Resource name must not exceed 100 characters";
            } else {
                oData.resourceNameState = "Success";
                oData.resourceNameStateText = "Valid resource name";
            }
            
            oCreateModel.setData(oData);
        },

        onCatalogTypeChange: function(oEvent) {
            var sValue = oEvent.getParameter("selectedItem").getKey();
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            if (sValue) {
                oData.catalogTypeState = "Success";
                oData.catalogTypeStateText = "Catalog type selected";
                
                // Smart suggestions based on catalog type
                switch (sValue) {
                    case "service":
                        oData.protocol = "https";
                        oData.authenticationMethod = "bearer";
                        oData.contentType = "json";
                        break;
                    case "api":
                        oData.protocol = "https";
                        oData.authenticationMethod = "oauth2";
                        oData.contentType = "json";
                        oData.searchable = true;
                        break;
                    case "data":
                        oData.dataClassification = "confidential";
                        oData.retentionPeriod = 730;
                        oData.compliance = ["gdpr"];
                        break;
                    case "security":
                        oData.visibility = "restricted";
                        oData.dataClassification = "restricted";
                        oData.authenticationMethod = "certificate";
                        break;
                }
            } else {
                oData.catalogTypeState = "Error";
                oData.catalogTypeStateText = "Please select a catalog type";
            }
            
            oCreateModel.setData(oData);
        },

        onResourceTypeChange: function(oEvent) {
            var sValue = oEvent.getParameter("selectedItem").getKey();
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            if (sValue) {
                oData.resourceTypeState = "Success";
                oData.resourceTypeStateText = "Resource type selected";
                
                // Adjust settings based on resource type
                switch (sValue) {
                    case "rest_api":
                    case "graphql":
                        oData.contentType = "json";
                        oData.protocol = "https";
                        break;
                    case "soap_service":
                        oData.contentType = "xml";
                        oData.protocol = "https";
                        break;
                    case "websocket":
                        oData.protocol = "websocket";
                        break;
                    case "database":
                        oData.protocol = "tcp";
                        oData.authenticationMethod = "basic";
                        break;
                }
            } else {
                oData.resourceTypeState = "Error";
                oData.resourceTypeStateText = "Please select a resource type";
            }
            
            oCreateModel.setData(oData);
        },

        onCategoryChange: function(oEvent) {
            var sValue = oEvent.getParameter("selectedItem").getKey();
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            if (sValue) {
                oData.categoryState = "Success";
                oData.categoryStateText = "Category selected";
            } else {
                oData.categoryState = "Error";
                oData.categoryStateText = "Please select a category";
            }
            
            oCreateModel.setData(oData);
        },

        onResourceUrlChange: function(oEvent) {
            var sValue = oEvent.getParameter("value");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            // Validate URL
            if (!sValue) {
                oData.resourceUrlState = "Error";
                oData.resourceUrlStateText = "Resource URL is required";
            } else {
                try {
                    new URL(sValue);
                    // Additional security validation
                    if (sValue.includes("javascript:") || sValue.includes("data:")) {
                        oData.resourceUrlState = "Error";
                        oData.resourceUrlStateText = "Invalid URL protocol";
                    } else {
                        oData.resourceUrlState = "Success";
                        oData.resourceUrlStateText = "Valid URL";
                        
                        // Auto-detect protocol
                        if (sValue.startsWith("https://")) {
                            oData.protocol = "https";
                        } else if (sValue.startsWith("http://")) {
                            oData.protocol = "http";
                        } else if (sValue.startsWith("ws://") || sValue.startsWith("wss://")) {
                            oData.protocol = "websocket";
                        }
                    }
                } catch (e) {
                    oData.resourceUrlState = "Error";
                    oData.resourceUrlStateText = "Invalid URL format";
                }
            }
            
            oCreateModel.setData(oData);
        },

        onCancelCreateCatalogEntry: function() {
            this._oCreateDialog.close();
            this._resetCreateModel();
        },

        onCreateDialogAfterOpen: function() {
            // Focus on first input field
            var oResourceNameInput = this.getView().byId("resourceNameInput");
            if (oResourceNameInput) {
                oResourceNameInput.focus();
            }
            
            // Subscribe to real-time validation updates
            this._startCreateValidation();
        },
        
        onCreateDialogAfterClose: function() {
            // Stop real-time validation
            this._stopCreateValidation();
        },
        
        _startCreateValidation: function() {
            // Real-time validation monitoring
            this._validationInterval = setInterval(() => {
                var oCreateModel = this.getView().getModel("create");
                if (oCreateModel) {
                    var oData = oCreateModel.getData();
                    this._updateCreateButtonState(oData);
                }
            }, 500);
        },
        
        _stopCreateValidation: function() {
            if (this._validationInterval) {
                clearInterval(this._validationInterval);
                this._validationInterval = null;
            }
        },
        
        _updateCreateButtonState: function(oData) {
            // Dynamic button enablement based on validation
            var bValid = oData.resourceName && oData.resourceName.length >= 3 &&
                        oData.catalogType && oData.resourceType && 
                        oData.category && oData.resourceUrl &&
                        oData.resourceNameState !== "Error" &&
                        oData.resourceUrlState !== "Error";
            
            var oCreateButton = this._oCreateDialog.getEndButton();
            if (oCreateButton) {
                oCreateButton.setEnabled(bValid);
            }
        },
        
        onAddMetadataProperty: function() {
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            if (!oData.metadataProperties) {
                oData.metadataProperties = [];
            }
            
            oData.metadataProperties.push({
                propertyName: "",
                propertyType: "string",
                propertyValue: "",
                isSearchable: false
            });
            
            oCreateModel.setData(oData);
        },
        
        onDeleteMetadataProperty: function(oEvent) {
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            var sPath = oEvent.getParameter("listItem").getBindingContext("create").getPath();
            var iIndex = parseInt(sPath.split("/").pop());
            
            oData.metadataProperties.splice(iIndex, 1);
            oCreateModel.setData(oData);
        },
        
        onComplianceChange: function(oEvent) {
            var aSelectedKeys = oEvent.getParameter("selectedKeys");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            // Auto-adjust settings based on compliance
            if (aSelectedKeys.includes("hipaa") || aSelectedKeys.includes("pci_dss")) {
                oData.encryptionRequired = true;
                oData.auditingEnabled = true;
                oData.dataClassification = "confidential";
            }
            
            if (aSelectedKeys.includes("gdpr")) {
                oData.retentionPeriod = Math.min(oData.retentionPeriod, 730); // Max 2 years
            }
            
            oCreateModel.setData(oData);
        },
        
        onDataClassificationChange: function(oEvent) {
            var sValue = oEvent.getParameter("selectedItem").getKey();
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            // Auto-adjust visibility based on classification
            switch (sValue) {
                case "top_secret":
                case "restricted":
                    oData.visibility = "restricted";
                    oData.searchable = false;
                    oData.autoDiscovery = false;
                    break;
                case "confidential":
                    oData.visibility = "internal";
                    break;
                case "public":
                    oData.visibility = "public";
                    oData.searchable = true;
                    break;
            }
            
            oCreateModel.setData(oData);
        },
        
        onDiscoveryFrequencyChange: function(oEvent) {
            var sValue = oEvent.getParameter("selectedItem").getKey();
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            // Enable/disable auto discovery based on frequency
            if (sValue === "manual") {
                oData.autoDiscovery = false;
            }
            
            oCreateModel.setData(oData);
        },
        
        onHealthCheckUrlChange: function(oEvent) {
            var sValue = oEvent.getParameter("value");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            // Validate health check URL
            if (sValue) {
                try {
                    new URL(sValue);
                    // Ensure it's related to resource URL
                    if (oData.resourceUrl && !sValue.startsWith(new URL(oData.resourceUrl).origin)) {
                        MessageToast.show(this.getResourceBundle().getText("warning.healthCheckDifferentDomain"));
                    }
                } catch (e) {
                    // Invalid URL, will be handled by main validation
                }
            }
            
            oCreateModel.setData(oData);
        },
        
        onSwaggerUrlChange: function(oEvent) {
            var sValue = oEvent.getParameter("value");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            // Auto-detect API documentation
            if (sValue && sValue.includes("swagger") || sValue.includes("openapi")) {
                oData.metadataSchema = "openapi";
            }
            
            oCreateModel.setData(oData);
        },
        
        onSearchableChange: function(oEvent) {
            var bState = oEvent.getParameter("state");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            // Reset search settings if not searchable
            if (!bState) {
                oData.searchTags = "";
                oData.searchWeight = 0;
            } else {
                oData.searchWeight = 50; // Default weight
            }
            
            oCreateModel.setData(oData);
        },
        
        onAutoDiscoveryChange: function(oEvent) {
            var bState = oEvent.getParameter("state");
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            // Set default discovery frequency
            if (bState && oData.discoveryFrequency === "manual") {
                oData.discoveryFrequency = "daily";
            }
            
            oCreateModel.setData(oData);
        },
        
        onConfirmCreateCatalogEntry: function() {
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            // Final validation
            if (!this._validateCreateData(oData)) {
                return;
            }
            
            this._oCreateDialog.setBusy(true);
            
            // Sanitize data for security
            var oSanitizedData = {
                resourceName: SecurityUtils.sanitizeCatalogData(oData.resourceName),
                description: SecurityUtils.sanitizeCatalogData(oData.description),
                catalogType: oData.catalogType,
                resourceType: oData.resourceType,
                category: oData.category,
                visibility: oData.visibility,
                resourceUrl: oData.resourceUrl,
                version: SecurityUtils.sanitizeCatalogData(oData.version),
                apiVersion: SecurityUtils.sanitizeCatalogData(oData.apiVersion),
                protocol: oData.protocol,
                authenticationMethod: oData.authenticationMethod,
                contentType: oData.contentType,
                documentation: oData.documentation,
                healthCheckUrl: oData.healthCheckUrl,
                swaggerUrl: oData.swaggerUrl,
                metadataSchema: oData.metadataSchema,
                keywords: SecurityUtils.sanitizeCatalogData(oData.keywords),
                tags: SecurityUtils.sanitizeCatalogData(oData.tags),
                owner: SecurityUtils.sanitizeCatalogData(oData.owner),
                maintainer: SecurityUtils.sanitizeCatalogData(oData.maintainer),
                license: oData.license,
                compliance: oData.compliance,
                dataClassification: oData.dataClassification,
                retentionPeriod: parseInt(oData.retentionPeriod) || 365,
                searchable: !!oData.searchable,
                searchTags: SecurityUtils.sanitizeCatalogData(oData.searchTags),
                searchWeight: parseInt(oData.searchWeight) || 50,
                autoDiscovery: !!oData.autoDiscovery,
                discoveryFrequency: oData.discoveryFrequency
            };
            
            SecurityUtils.secureCallFunction(this.getView().getModel(), "/CreateCatalogEntry", {
                urlParameters: oSanitizedData,
                success: function(data) {
                    this._oCreateDialog.setBusy(false);
                    this._oCreateDialog.close();
                    MessageToast.show(this.getResourceBundle().getText("msg.catalogEntryCreated"));
                    this._refreshEntryData();
                    this._resetCreateModel();
                }.bind(this),
                error: function(error) {
                    this._oCreateDialog.setBusy(false);
                    var errorMsg = SecurityUtils.escapeHTML(error.message || "Unknown error");
                    MessageBox.error(this.getResourceBundle().getText("error.createEntryFailed") + ": " + errorMsg);
                }.bind(this)
            });
        },

        _validateCreateData: function(oData) {
            if (!oData.resourceName || oData.resourceName.length < 3) {
                MessageBox.error(this.getResourceBundle().getText("validation.resourceNameRequired"));
                return false;
            }
            
            if (!oData.catalogType) {
                MessageBox.error(this.getResourceBundle().getText("validation.catalogTypeRequired"));
                return false;
            }
            
            if (!oData.resourceType) {
                MessageBox.error(this.getResourceBundle().getText("validation.resourceTypeRequired"));
                return false;
            }
            
            if (!oData.category) {
                MessageBox.error(this.getResourceBundle().getText("validation.categoryRequired"));
                return false;
            }
            
            if (!oData.resourceUrl) {
                MessageBox.error(this.getResourceBundle().getText("validation.resourceUrlRequired"));
                return false;
            }
            
            return true;
        },

        _resetCreateModel: function() {
            var oCreateModel = this.getView().getModel("create");
            var oData = oCreateModel.getData();
            
            oData.resourceName = "";
            oData.description = "";
            oData.catalogType = "";
            oData.resourceType = "";
            oData.category = "";
            oData.visibility = "internal";
            oData.resourceNameState = "";
            oData.resourceNameStateText = "";
            oData.catalogTypeState = "";
            oData.catalogTypeStateText = "";
            oData.resourceTypeState = "";
            oData.resourceTypeStateText = "";
            oData.categoryState = "";
            oData.categoryStateText = "";
            oData.resourceUrl = "";
            oData.resourceUrlState = "";
            oData.resourceUrlStateText = "";
            oData.version = "";
            oData.apiVersion = "";
            oData.protocol = "https";
            oData.authenticationMethod = "bearer";
            oData.contentType = "json";
            oData.documentation = "";
            oData.healthCheckUrl = "";
            oData.swaggerUrl = "";
            oData.metadataSchema = "dublincore";
            oData.keywords = "";
            oData.tags = "";
            oData.owner = "";
            oData.maintainer = "";
            oData.license = "";
            oData.compliance = [];
            oData.dataClassification = "internal";
            oData.retentionPeriod = 365;
            oData.searchable = true;
            oData.searchTags = "";
            oData.searchWeight = 50;
            oData.autoDiscovery = false;
            oData.discoveryFrequency = "daily";
            oData.metadataProperties = [];
            oData.encryptionRequired = false;
            oData.auditingEnabled = false;
            oData.complianceScore = 0;
            oData.sensitiveDataDetected = false;
            
            oCreateModel.setData(oData);
        },
        

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