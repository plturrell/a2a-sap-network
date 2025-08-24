sap.ui.define([
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel",
    "sap/m/Dialog",
    "sap/m/Button",
    "sap/m/Text",
    "a2a/ext/agent0/utils/SecurityUtils"
], (MessageToast, MessageBox, Fragment, JSONModel, Dialog, Button, Text, SecurityUtils) => {
    "use strict";

    return {
        /**
         * Called when controller is initialized
         */
        onInit() {
            this._initializeModels();
            this._attachRouteMatched();
        },

        /**
         * Initialize view models
         */
        _initializeModels() {
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
        _attachRouteMatched() {
            const oRouter = this.getOwnerComponent().getRouter();
            oRouter.getRoute("object").attachPatternMatched(this._onObjectMatched, this);
        },

        /**
         * Handle route matched
         */
        _onObjectMatched(oEvent) {
            const sProductId = oEvent.getParameter("arguments").productId;
            this._loadAdditionalData(sProductId);
        },

        /**
         * Edit Dublin Core metadata
         */
        onEditMetadata() {
            const oView = this.getView();

            if (!this._oMetadataDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.ext.agent0.fragment.DublinCoreEditor",
                    controller: this
                }).then((oDialog) => {
                    this._oMetadataDialog = oDialog;
                    oView.addDependent(this._oMetadataDialog);
                    this._prepareMetadataEditor();
                    this._oMetadataDialog.open();
                });
            } else {
                this._prepareMetadataEditor();
                this._oMetadataDialog.open();
            }
        },

        /**
         * Validate schema
         */
        onValidateSchema() {
            const oBinding = this.getView().getBindingContext();
            const sPath = oBinding.getPath();

            const oResourceBundle = this.getView().getModel("i18n").getResourceBundle();
            MessageToast.show(oResourceBundle.getText("msg.validatingSchema"));

            const oModel = this.getView().getModel();
            const oData = oBinding.getObject();
            const validation = SecurityUtils.validateSchema(oData.schema);

            if (!validation.valid) {
                const oResourceBundle = this.getView().getModel("i18n").getResourceBundle();
                MessageBox.error(`${oResourceBundle.getText("error.validationFailed") }: ${ validation.error}`);
                return;
            }

            SecurityUtils.secureCallFunction(oModel, `${sPath}/validateSchema`, validation.sanitized,
                (oResult) => {
                    const oValidationResult = oResult.getBoundContext().getObject();
                    const oResourceBundle = this.getView().getModel("i18n").getResourceBundle();

                    if (oValidationResult.isValid) {
                        MessageBox.success(oResourceBundle.getText("msg.validationPassed"));
                    } else {
                        this._showValidationErrors(oValidationResult.errors);
                    }
                },
                (error) => {
                    const oResourceBundle = this.getView().getModel("i18n").getResourceBundle();
                    MessageBox.error(oResourceBundle.getText("error.validationFailed"));
                }
            );
        },

        /**
         * Generate Dublin Core metadata
         */
        onGenerateDublinCore() {
            const oResourceBundle = this.getView().getModel("i18n").getResourceBundle();
            MessageBox.confirm(oResourceBundle.getText("msg.confirmGenerateDublinCore"), {
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        this._generateDublinCore();
                    }
                }.bind(this)
            });
        },

        /**
         * View data lineage
         */
        onViewLineage() {
            const oView = this.getView();

            if (!this._oLineageDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.ext.agent0.fragment.DataLineageViewer",
                    controller: this
                }).then((oDialog) => {
                    this._oLineageDialog = oDialog;
                    oView.addDependent(this._oLineageDialog);
                    this._loadLineageData();
                    this._oLineageDialog.open();
                });
            } else {
                this._loadLineageData();
                this._oLineageDialog.open();
            }
        },

        /**
         * Assess quality
         */
        onAssessQuality() {
            const oBinding = this.getView().getBindingContext();
            const sPath = oBinding.getPath();

            const oResourceBundle = this.getView().getModel("i18n").getResourceBundle();
            MessageToast.show(oResourceBundle.getText("msg.assessingQuality"));

            const oModel = this.getView().getModel();
            const oData = oBinding.getObject();
            const validation = SecurityUtils.validateQualityMetrics(oData.qualityMetrics || {});

            SecurityUtils.secureCallFunction(oModel, `${sPath}/assessQuality`, validation.sanitized || {},
                (oResult) => {
                    const oQualityResult = oResult.getBoundContext().getObject();
                    this._showQualityResults(oQualityResult);
                },
                (error) => {
                    const oResourceBundle = this.getView().getModel("i18n").getResourceBundle();
                    MessageBox.error(oResourceBundle.getText("error.qualityAssessmentFailed"));
                }
            );
        },

        /**
         * Publish to catalog
         */
        onPublishToCatalog() {
            const oResourceBundle = this.getView().getModel("i18n").getResourceBundle();
            MessageBox.confirm(oResourceBundle.getText("msg.confirmPublish"), {
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        this._publishProduct();
                    }
                }.bind(this)
            });
        },

        /**
         * Create new version
         */
        onCreateVersion() {
            const oView = this.getView();

            if (!this._oVersionDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.ext.agent0.fragment.CreateVersion",
                    controller: this
                }).then((oDialog) => {
                    this._oVersionDialog = oDialog;
                    oView.addDependent(this._oVersionDialog);
                    this._oVersionDialog.open();
                });
            } else {
                this._oVersionDialog.open();
            }
        },

        /**
         * Compare versions
         */
        onCompareVersions() {
            const oView = this.getView();

            if (!this._oCompareDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.ext.agent0.fragment.VersionComparison",
                    controller: this
                }).then((oDialog) => {
                    this._oCompareDialog = oDialog;
                    oView.addDependent(this._oCompareDialog);
                    this._loadVersions();
                    this._oCompareDialog.open();
                });
            } else {
                this._loadVersions();
                this._oCompareDialog.open();
            }
        },

        /**
         * Load additional data for the object page
         * @private
         */
        _loadAdditionalData(sProductId) {
            // Load lineage, quality metrics, versions etc.
            const oViewModel = this.getView().getModel("objectView");
            oViewModel.setProperty("/busy", true);

            Promise.all([
                this._fetchLineageData(sProductId),
                this._fetchQualityMetrics(sProductId),
                this._fetchVersionHistory(sProductId)
            ]).then((aResults) => {
                oViewModel.setProperty("/lineageData", aResults[0]);
                oViewModel.setProperty("/qualityMetrics", aResults[1]);
                oViewModel.setProperty("/versions", aResults[2]);
            }).finally(() => {
                oViewModel.setProperty("/busy", false);
            });
        },

        /**
         * Prepare metadata editor
         * @private
         */
        _prepareMetadataEditor() {
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
        async _generateDublinCore() {
            const oBinding = this.getView().getBindingContext();
            const sPath = oBinding.getPath();

            try {
                const oModel = this.getView().getModel();
                const oData = oBinding.getObject();
                const validation = SecurityUtils.validateMetadata(oData);

                if (!validation.valid) {
                    MessageBox.error(validation.error);
                    return;
                }

                await SecurityUtils.secureCallFunction(oModel, `${sPath}/generateDublinCore`, validation.sanitized,
                    () => { /* success handled below */ },
                    () => { throw new Error("Generation failed"); }
                );

                const oResourceBundle = this.getView().getModel("i18n").getResourceBundle();
                MessageBox.success(oResourceBundle.getText("msg.dublinCoreGenerated"));

                // Refresh the binding
                oBinding.refresh();
            } catch (error) {
                const oResourceBundle = this.getView().getModel("i18n").getResourceBundle();
                MessageBox.error(oResourceBundle.getText("error.dublinCoreGenerationFailed"));
            }
        },

        /**
         * Show validation errors
         * @private
         */
        _showValidationErrors(aErrors) {
            const oView = this.getView();

            if (!this._oErrorDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.ext.agent0.fragment.ValidationErrors",
                    controller: this
                }).then((oDialog) => {
                    this._oErrorDialog = oDialog;
                    oView.addDependent(this._oErrorDialog);

                    const oErrorModel = new JSONModel({ errors: aErrors });
                    this._oErrorDialog.setModel(oErrorModel, "errors");
                    this._oErrorDialog.open();
                });
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
        async _loadLineageData() {
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
        _fetchLineageData(sProductId) {
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
        _fetchQualityMetrics(sProductId) {
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
        _fetchVersionHistory(sProductId) {
            // Simulate version history - replace with actual service call
            return Promise.resolve([
                { version: "2.0", date: new Date(), author: "John Doe", changes: "Added new fields" },
                { version: "1.5", date: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), author: "Jane Smith", changes: "Performance improvements" },
                { version: "1.0", date: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000), author: "John Doe", changes: "Initial release" }
            ]);
        },

        /**
         * Show quality results
         * @private
         */
        _showQualityResults(oQualityResult) {
            const oView = this.getView();

            if (!this._oQualityDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.ext.agent0.fragment.QualityResults",
                    controller: this
                }).then((oDialog) => {
                    this._oQualityDialog = oDialog;
                    oView.addDependent(this._oQualityDialog);

                    const oQualityModel = new JSONModel(oQualityResult);
                    this._oQualityDialog.setModel(oQualityModel, "quality");
                    this._oQualityDialog.open();
                });
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
        async _publishProduct() {
            const oBinding = this.getView().getBindingContext();
            const sPath = oBinding.getPath();

            try {
                const oModel = this.getView().getModel();
                const oData = oBinding.getObject();
                const validation = SecurityUtils.validateDublinCore(oData.dublinCore);

                if (!validation.valid) {
                    MessageBox.error(validation.error);
                    return;
                }

                await SecurityUtils.secureCallFunction(oModel, `${sPath}/publish`, validation.sanitized,
                    () => { /* success handled below */ },
                    () => { throw new Error("Publish failed"); }
                );

                const oResourceBundle = this.getView().getModel("i18n").getResourceBundle();
                MessageBox.success(oResourceBundle.getText("msg.publishSuccess"));

                // Refresh the binding
                oBinding.refresh();
            } catch (error) {
                const oResourceBundle = this.getView().getModel("i18n").getResourceBundle();
                MessageBox.error(oResourceBundle.getText("error.publishFailed"));
            }
        },

        /**
         * Load versions for comparison
         * @private
         */
        async _loadVersions() {
            const oBinding = this.getView().getBindingContext();
            const sProductId = oBinding.getObject().ID;

            const aVersions = await this._fetchVersionHistory(sProductId);
            const oViewModel = this.getView().getModel("objectView");
            oViewModel.setProperty("/versions", aVersions);
        },

        /**
         * Refresh dashboard data
         * @public
         */
        onRefreshDashboard() {
            const oBinding = this.getView().getBindingContext();
            if (!oBinding) {
                MessageToast.show("No data product selected");
                return;
            }

            const sProductId = oBinding.getObject().ID;
            const oViewModel = this.getView().getModel("objectView");

            oViewModel.setProperty("/busy", true);
            MessageToast.show("Refreshing dashboard data...");

            Promise.all([
                this._fetchLineageData(sProductId),
                this._fetchQualityMetrics(sProductId),
                this._fetchVersionHistory(sProductId)
            ]).then(function(aResults) {
                oViewModel.setProperty("/lineageData", aResults[0]);
                oViewModel.setProperty("/qualityMetrics", aResults[1]);
                oViewModel.setProperty("/versions", aResults[2]);

                const oResourceBundle = this.getView().getModel("i18n").getResourceBundle();
                MessageToast.show(oResourceBundle.getText("msg.dashboardRefreshed") || "Dashboard refreshed successfully");

                // Refresh the binding to get latest data
                oBinding.refresh();
            }).catch(function(error) {
                const oResourceBundle = this.getView().getModel("i18n").getResourceBundle();
                MessageBox.error(oResourceBundle.getText("error.dashboardRefreshFailed") || "Failed to refresh dashboard data");
            }).finally(() => {
                oViewModel.setProperty("/busy", false);
            });
        },

        /**
         * Export dashboard data
         * @public
         */
        onExportDashboard() {
            const oBinding = this.getView().getBindingContext();
            if (!oBinding) {
                MessageToast.show("No data product selected");
                return;
            }

            const oProductData = oBinding.getObject();
            const oViewModel = this.getView().getModel("objectView");
            const oViewData = oViewModel.getData();

            // Prepare export data
            const oExportData = {
                product: {
                    id: oProductData.ID,
                    title: oProductData.title,
                    description: oProductData.description,
                    createdAt: oProductData.createdAt,
                    modifiedAt: oProductData.modifiedAt
                },
                lineage: oViewData.lineageData,
                qualityMetrics: oViewData.qualityMetrics,
                versions: oViewData.versions,
                dublinCore: oProductData.dublinCore,
                exportedAt: new Date().toISOString(),
                exportedBy: this.getOwnerComponent().getModel("user")?.getProperty("/name") || "Unknown"
            };

            // Validate export data for security
            const validation = SecurityUtils.validateExportData ? SecurityUtils.validateExportData(oExportData) : { valid: true, sanitized: oExportData };

            if (!validation.valid) {
                MessageBox.error(`Export data validation failed: ${ validation.error}`);
                return;
            }

            try {
                // Create and download JSON file
                const sJsonData = JSON.stringify(validation.sanitized, null, 2);
                const oBlob = new Blob([sJsonData], { type: "application/json" });
                const sUrl = URL.createObjectURL(oBlob);

                const oLink = document.createElement("a");
                oLink.href = sUrl;
                oLink.download = `data-product-${oProductData.ID}-dashboard-${new Date().toISOString().split("T")[0]}.json`;
                document.body.appendChild(oLink);
                oLink.click();
                document.body.removeChild(oLink);
                URL.revokeObjectURL(sUrl);

                const oResourceBundle = this.getView().getModel("i18n").getResourceBundle();
                MessageToast.show(oResourceBundle.getText("msg.dashboardExported") || "Dashboard data exported successfully");

            } catch (error) {
                const oResourceBundle = this.getView().getModel("i18n").getResourceBundle();
                MessageBox.error(oResourceBundle.getText("error.dashboardExportFailed") || "Failed to export dashboard data");
            }
        },

        /**
         * Complete wizard process
         * @public
         */
        onWizardComplete() {
            const oView = this.getView();
            const oWizardModel = oView.getModel("wizard");

            if (!oWizardModel) {
                MessageBox.error("Wizard data not found");
                return;
            }

            const oWizardData = oWizardModel.getData();

            // Validate wizard data
            const validation = SecurityUtils.validateWizardData ? SecurityUtils.validateWizardData(oWizardData) : { valid: true, sanitized: oWizardData };

            if (!validation.valid) {
                MessageBox.error(`Wizard validation failed: ${ validation.error}`);
                return;
            }

            MessageBox.confirm("Complete the data product creation process?", {
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        this._processWizardCompletion(validation.sanitized);
                    }
                }.bind(this)
            });
        },

        /**
         * Process wizard completion
         * @private
         */
        async _processWizardCompletion(oWizardData) {
            const oModel = this.getView().getModel();
            const oView = this.getView();

            oView.setBusy(true);

            try {
                // Create new data product
                const oNewProduct = {
                    title: oWizardData.basicInfo?.title || "",
                    description: oWizardData.basicInfo?.description || "",
                    format: oWizardData.basicInfo?.format || "",
                    schema: oWizardData.schema || {},
                    dublinCore: oWizardData.metadata || {},
                    qualityRules: oWizardData.qualityRules || [],
                    transformationRules: oWizardData.transformationRules || [],
                    status: "DRAFT",
                    createdAt: new Date().toISOString()
                };

                // Call backend service to create product
                await SecurityUtils.secureCallFunction(oModel, "/DataProducts", oNewProduct,
                    (oResult) => {
                        const oCreatedProduct = oResult.getBoundContext().getObject();
                        const oResourceBundle = oView.getModel("i18n").getResourceBundle();

                        MessageBox.success(oResourceBundle.getText("msg.productCreated") || "Data product created successfully", {
                            onClose: function() {
                                // Navigate to the created product
                                const oRouter = this.getOwnerComponent().getRouter();
                                oRouter.navTo("object", {
                                    productId: oCreatedProduct.ID
                                });

                                // Close wizard dialog if it exists
                                if (this._oWizardDialog) {
                                    this._oWizardDialog.close();
                                }
                            }.bind(this)
                        });
                    },
                    (error) => {
                        throw new Error(`Product creation failed: ${ error.message}`);
                    }
                );

            } catch (error) {
                const oResourceBundle = oView.getModel("i18n").getResourceBundle();
                MessageBox.error(oResourceBundle.getText("error.productCreationFailed") || `Failed to create data product: ${ error.message}`);
            } finally {
                oView.setBusy(false);
            }
        },

        /**
         * Cancel wizard
         * @public
         */
        onWizardCancel() {
            MessageBox.confirm("Cancel the data product creation process? All entered data will be lost.", {
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        // Reset wizard model
                        const oView = this.getView();
                        const oWizardModel = new JSONModel({
                            basicInfo: {},
                            metadata: {},
                            schema: {},
                            qualityRules: [],
                            transformationRules: []
                        });
                        oView.setModel(oWizardModel, "wizard");

                        // Close wizard dialog if it exists
                        if (this._oWizardDialog) {
                            this._oWizardDialog.close();
                        }

                        MessageToast.show("Wizard cancelled");
                    }
                }.bind(this)
            });
        },

        /**
         * Close dashboard dialog
         * @public
         */
        onCloseDashboard() {
            if (this._oDashboardDialog) {
                this._oDashboardDialog.close();
            }
        }
    };
});