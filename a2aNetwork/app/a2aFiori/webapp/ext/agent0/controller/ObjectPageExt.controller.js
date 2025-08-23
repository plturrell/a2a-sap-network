sap.ui.define([
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel",
    "sap/m/Dialog",
    "sap/m/Button",
    "sap/m/Text"
], function (MessageToast, MessageBox, Fragment, JSONModel, Dialog, Button, Text) {
    'use strict';

    return {
        /**
         * Called when controller is initialized
         */
        onInit: function () {
            this._initializeModels();
            this._attachRouteMatched();
        },

        /**
         * Initialize view models
         */
        _initializeModels: function () {
            const oViewModel = new JSONModel({
                busy: false,
                editable: false,
                lineageData: null,
                qualityMetrics: null,
                versions: []
            });
            this.getView().setModel(oViewModel, "objectView");
        },

        /**
         * Attach to route matched event
         */
        _attachRouteMatched: function () {
            const oRouter = this.getOwnerComponent().getRouter();
            oRouter.getRoute("object").attachPatternMatched(this._onObjectMatched, this);
        },

        /**
         * Handle route matched
         */
        _onObjectMatched: function (oEvent) {
            const sProductId = oEvent.getParameter("arguments").productId;
            this._loadAdditionalData(sProductId);
        },

        /**
         * Edit Dublin Core metadata
         */
        onEditMetadata: function () {
            const oView = this.getView();
            
            if (!this._oMetadataDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.ext.agent0.fragment.DublinCoreEditor",
                    controller: this
                }).then(function (oDialog) {
                    this._oMetadataDialog = oDialog;
                    oView.addDependent(this._oMetadataDialog);
                    this._prepareMetadataEditor();
                    this._oMetadataDialog.open();
                }.bind(this));
            } else {
                this._prepareMetadataEditor();
                this._oMetadataDialog.open();
            }
        },

        /**
         * Validate schema
         */
        onValidateSchema: function () {
            const oBinding = this.getView().getBindingContext();
            const sPath = oBinding.getPath();
            
            MessageToast.show("Validating schema...");
            
            const oModel = this.getView().getModel();
            oModel.bindContext(`${sPath}/validateSchema(...)`).execute()
                .then(function (oResult) {
                    const oValidationResult = oResult.getBoundContext().getObject();
                    if (oValidationResult.isValid) {
                        MessageBox.success("Schema validation passed!");
                    } else {
                        this._showValidationErrors(oValidationResult.errors);
                    }
                }.bind(this))
                .catch(function (error) {
                    MessageBox.error("Schema validation failed");
                });
        },

        /**
         * Generate Dublin Core metadata
         */
        onGenerateDublinCore: function () {
            MessageBox.confirm("Generate Dublin Core metadata from existing fields?", {
                onClose: function (oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        this._generateDublinCore();
                    }
                }.bind(this)
            });
        },

        /**
         * View data lineage
         */
        onViewLineage: function () {
            const oView = this.getView();
            
            if (!this._oLineageDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.ext.agent0.fragment.DataLineageViewer",
                    controller: this
                }).then(function (oDialog) {
                    this._oLineageDialog = oDialog;
                    oView.addDependent(this._oLineageDialog);
                    this._loadLineageData();
                    this._oLineageDialog.open();
                }.bind(this));
            } else {
                this._loadLineageData();
                this._oLineageDialog.open();
            }
        },

        /**
         * Assess quality
         */
        onAssessQuality: function () {
            const oBinding = this.getView().getBindingContext();
            const sPath = oBinding.getPath();
            
            MessageToast.show("Assessing data quality...");
            
            const oModel = this.getView().getModel();
            oModel.bindContext(`${sPath}/assessQuality(...)`).execute()
                .then(function (oResult) {
                    const oQualityResult = oResult.getBoundContext().getObject();
                    this._showQualityResults(oQualityResult);
                }.bind(this))
                .catch(function (error) {
                    MessageBox.error("Quality assessment failed");
                });
        },

        /**
         * Publish to catalog
         */
        onPublishToCatalog: function () {
            MessageBox.confirm("Publish this data product to the catalog?", {
                onClose: function (oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        this._publishProduct();
                    }
                }.bind(this)
            });
        },

        /**
         * Create new version
         */
        onCreateVersion: function () {
            const oView = this.getView();
            
            if (!this._oVersionDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.ext.agent0.fragment.CreateVersion",
                    controller: this
                }).then(function (oDialog) {
                    this._oVersionDialog = oDialog;
                    oView.addDependent(this._oVersionDialog);
                    this._oVersionDialog.open();
                }.bind(this));
            } else {
                this._oVersionDialog.open();
            }
        },

        /**
         * Compare versions
         */
        onCompareVersions: function () {
            const oView = this.getView();
            
            if (!this._oCompareDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.ext.agent0.fragment.VersionComparison",
                    controller: this
                }).then(function (oDialog) {
                    this._oCompareDialog = oDialog;
                    oView.addDependent(this._oCompareDialog);
                    this._loadVersions();
                    this._oCompareDialog.open();
                }.bind(this));
            } else {
                this._loadVersions();
                this._oCompareDialog.open();
            }
        },

        /**
         * Load additional data for the object page
         * @private
         */
        _loadAdditionalData: function (sProductId) {
            // Load lineage, quality metrics, versions etc.
            const oViewModel = this.getView().getModel("objectView");
            oViewModel.setProperty("/busy", true);
            
            Promise.all([
                this._fetchLineageData(sProductId),
                this._fetchQualityMetrics(sProductId),
                this._fetchVersionHistory(sProductId)
            ]).then(function (aResults) {
                oViewModel.setProperty("/lineageData", aResults[0]);
                oViewModel.setProperty("/qualityMetrics", aResults[1]);
                oViewModel.setProperty("/versions", aResults[2]);
            }).finally(function () {
                oViewModel.setProperty("/busy", false);
            });
        },

        /**
         * Prepare metadata editor
         * @private
         */
        _prepareMetadataEditor: function () {
            const oBinding = this.getView().getBindingContext();
            const oProduct = oBinding.getObject();
            
            const oEditorModel = new JSONModel({
                title: oProduct.title || "",
                creator: oProduct.creator || "",
                subject: oProduct.subject || "",
                description: oProduct.description || "",
                publisher: oProduct.publisher || "",
                contributor: oProduct.contributor || "",
                date: oProduct.date || new Date(),
                type: oProduct.type || "",
                format: oProduct.format || "",
                identifier: oProduct.identifier || "",
                source: oProduct.source || "",
                language: oProduct.language || "",
                relation: oProduct.relation || "",
                coverage: oProduct.coverage || "",
                rights: oProduct.rights || ""
            });
            
            this._oMetadataDialog.setModel(oEditorModel, "editor");
        },

        /**
         * Generate Dublin Core metadata
         * @private
         */
        _generateDublinCore: async function () {
            const oBinding = this.getView().getBindingContext();
            const sPath = oBinding.getPath();
            
            try {
                const oModel = this.getView().getModel();
                const oResult = await oModel.bindContext(`${sPath}/generateDublinCore(...)`).execute();
                
                MessageBox.success("Dublin Core metadata generated successfully");
                
                // Refresh the binding
                oBinding.refresh();
            } catch (error) {
                MessageBox.error("Failed to generate Dublin Core metadata");
            }
        },

        /**
         * Show validation errors
         * @private
         */
        _showValidationErrors: function (aErrors) {
            const oView = this.getView();
            
            if (!this._oErrorDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.ext.agent0.fragment.ValidationErrors",
                    controller: this
                }).then(function (oDialog) {
                    this._oErrorDialog = oDialog;
                    oView.addDependent(this._oErrorDialog);
                    
                    const oErrorModel = new JSONModel({ errors: aErrors });
                    this._oErrorDialog.setModel(oErrorModel, "errors");
                    this._oErrorDialog.open();
                }.bind(this));
            } else {
                const oErrorModel = new JSONModel({ errors: aErrors });
                this._oErrorDialog.setModel(oErrorModel, "errors");
                this._oErrorDialog.open();
            }
        },

        /**
         * Load lineage data
         * @private
         */
        _loadLineageData: async function () {
            const oBinding = this.getView().getBindingContext();
            const sProductId = oBinding.getObject().ID;
            
            const oLineageData = await this._fetchLineageData(sProductId);
            const oViewModel = this.getView().getModel("objectView");
            oViewModel.setProperty("/lineageData", oLineageData);
        },

        /**
         * Fetch lineage data from backend
         * @private
         */
        _fetchLineageData: function (sProductId) {
            // Simulate lineage data - replace with actual service call
            return Promise.resolve({
                sources: [
                    { id: "src1", name: "Customer Database", type: "database" },
                    { id: "src2", name: "Sales System", type: "api" }
                ],
                transformations: [
                    { id: "t1", name: "Data Cleansing", type: "quality" },
                    { id: "t2", name: "Aggregation", type: "transform" }
                ],
                consumers: [
                    { id: "c1", name: "Analytics Dashboard", type: "application" },
                    { id: "c2", name: "ML Model", type: "model" }
                ]
            });
        },

        /**
         * Fetch quality metrics
         * @private
         */
        _fetchQualityMetrics: function (sProductId) {
            // Simulate quality metrics - replace with actual service call
            return Promise.resolve({
                completeness: 95,
                accuracy: 88,
                consistency: 92,
                timeliness: 85,
                validity: 90,
                uniqueness: 98
            });
        },

        /**
         * Fetch version history
         * @private
         */
        _fetchVersionHistory: function (sProductId) {
            // Simulate version history - replace with actual service call
            return Promise.resolve([
                { version: "2.0", date: new Date(), author: "John Doe", changes: "Added new fields" },
                { version: "1.5", date: new Date(Date.now() - 30*24*60*60*1000), author: "Jane Smith", changes: "Performance improvements" },
                { version: "1.0", date: new Date(Date.now() - 90*24*60*60*1000), author: "John Doe", changes: "Initial release" }
            ]);
        },

        /**
         * Show quality results
         * @private
         */
        _showQualityResults: function (oQualityResult) {
            const oView = this.getView();
            
            if (!this._oQualityDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.ext.agent0.fragment.QualityResults",
                    controller: this
                }).then(function (oDialog) {
                    this._oQualityDialog = oDialog;
                    oView.addDependent(this._oQualityDialog);
                    
                    const oQualityModel = new JSONModel(oQualityResult);
                    this._oQualityDialog.setModel(oQualityModel, "quality");
                    this._oQualityDialog.open();
                }.bind(this));
            } else {
                const oQualityModel = new JSONModel(oQualityResult);
                this._oQualityDialog.setModel(oQualityModel, "quality");
                this._oQualityDialog.open();
            }
        },

        /**
         * Publish product
         * @private
         */
        _publishProduct: async function () {
            const oBinding = this.getView().getBindingContext();
            const sPath = oBinding.getPath();
            
            try {
                const oModel = this.getView().getModel();
                await oModel.bindContext(`${sPath}/publish(...)`).execute();
                
                MessageBox.success("Product published successfully");
                
                // Refresh the binding
                oBinding.refresh();
            } catch (error) {
                MessageBox.error("Failed to publish product");
            }
        },

        /**
         * Load versions for comparison
         * @private
         */
        _loadVersions: async function () {
            const oBinding = this.getView().getBindingContext();
            const sProductId = oBinding.getObject().ID;
            
            const aVersions = await this._fetchVersionHistory(sProductId);
            const oViewModel = this.getView().getModel("objectView");
            oViewModel.setProperty("/versions", aVersions);
        }
    };
});