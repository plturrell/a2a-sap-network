sap.ui.define([
    "sap/m/MessageToast",
    "sap/m/Dialog",
    "sap/m/Button",
    "sap/m/Text",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel"
], function (MessageToast, Dialog, Button, Text, Fragment, JSONModel) {
    'use strict';

    return {
        /**
         * Called when controller is initialized
         */
        onInit: function () {
            this._initializeModels();
        },

        /**
         * Initialize models for the view
         */
        _initializeModels: function () {
            const oViewModel = new JSONModel({
                busy: false,
                dashboardData: {
                    totalProducts: 0,
                    activeProducts: 0,
                    averageQuality: 0,
                    recentActivity: []
                }
            });
            this.getView().setModel(oViewModel, "viewModel");
        },

        /**
         * Handler for Create Data Product action
         */
        onCreateDataProduct: function () {
            const oView = this.getView();
            
            // Create dialog lazily
            if (!this._oCreateDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.ext.agent0.fragment.CreateDataProduct",
                    controller: this
                }).then(function (oDialog) {
                    this._oCreateDialog = oDialog;
                    oView.addDependent(this._oCreateDialog);
                    this._oCreateDialog.open();
                }.bind(this));
            } else {
                this._oCreateDialog.open();
            }
        },

        /**
         * Handler for Import Metadata action
         */
        onImportMetadata: function () {
            const oView = this.getView();
            
            if (!this._oImportDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.ext.agent0.fragment.ImportMetadata",
                    controller: this
                }).then(function (oDialog) {
                    this._oImportDialog = oDialog;
                    oView.addDependent(this._oImportDialog);
                    this._oImportDialog.open();
                }.bind(this));
            } else {
                this._oImportDialog.open();
            }
        },

        /**
         * Handler for Dashboard action
         */
        onOpenDashboard: function () {
            const oView = this.getView();
            
            if (!this._oDashboardDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.ext.agent0.fragment.DataProductDashboard",
                    controller: this
                }).then(function (oDialog) {
                    this._oDashboardDialog = oDialog;
                    oView.addDependent(this._oDashboardDialog);
                    this._loadDashboardData();
                    this._oDashboardDialog.open();
                }.bind(this));
            } else {
                this._loadDashboardData();
                this._oDashboardDialog.open();
            }
        },

        /**
         * Handler for Validate Metadata action
         */
        onValidateMetadata: function (oEvent) {
            const oTable = this.byId("DataProductsTable");
            const aSelectedContexts = oTable.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageToast.show("Please select at least one data product");
                return;
            }
            
            this._validateSelectedProducts(aSelectedContexts);
        },

        /**
         * Handler for Bulk Update action
         */
        onBulkUpdate: function () {
            const oTable = this.byId("DataProductsTable");
            const aSelectedContexts = oTable.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageToast.show("Please select products to update");
                return;
            }
            
            const oView = this.getView();
            
            if (!this._oBulkUpdateDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.ext.agent0.fragment.BulkUpdate",
                    controller: this
                }).then(function (oDialog) {
                    this._oBulkUpdateDialog = oDialog;
                    oView.addDependent(this._oBulkUpdateDialog);
                    this._oBulkUpdateDialog.open();
                }.bind(this));
            } else {
                this._oBulkUpdateDialog.open();
            }
        },

        /**
         * Handler for Export Catalog action
         */
        onExportCatalog: function () {
            const oDialog = new Dialog({
                title: "Export Data Product Catalog",
                type: "Message",
                content: [
                    new Text({ text: "Select export format:" }),
                    new sap.m.RadioButtonGroup({
                        columns: 1,
                        selectedIndex: 0,
                        buttons: [
                            new sap.m.RadioButton({ text: "JSON" }),
                            new sap.m.RadioButton({ text: "CSV" }),
                            new sap.m.RadioButton({ text: "XML (Dublin Core)" })
                        ]
                    })
                ],
                beginButton: new Button({
                    text: "Export",
                    type: "Emphasized",
                    press: function () {
                        const iSelectedIndex = oDialog.getContent()[1].getSelectedIndex();
                        const aFormats = ["json", "csv", "xml"];
                        this._exportCatalog(aFormats[iSelectedIndex]);
                        oDialog.close();
                    }.bind(this)
                }),
                endButton: new Button({
                    text: "Cancel",
                    press: function () {
                        oDialog.close();
                    }
                }),
                afterClose: function () {
                    oDialog.destroy();
                }
            });
            
            oDialog.open();
        },

        /**
         * Load dashboard data
         * @private
         */
        _loadDashboardData: async function () {
            const oViewModel = this.getView().getModel("viewModel");
            oViewModel.setProperty("/busy", true);
            
            try {
                // Call backend service to get dashboard data
                const oModel = this.getView().getModel();
                const aDashboardData = await this._fetchDashboardData(oModel);
                
                oViewModel.setProperty("/dashboardData", aDashboardData);
            } catch (error) {
                MessageToast.show("Failed to load dashboard data");
            } finally {
                oViewModel.setProperty("/busy", false);
            }
        },

        /**
         * Fetch dashboard data from backend
         * @private
         */
        _fetchDashboardData: function (oModel) {
            // Simulate dashboard data - replace with actual service call
            return Promise.resolve({
                totalProducts: 150,
                activeProducts: 120,
                averageQuality: 85.5,
                recentActivity: [
                    { action: "Created", product: "Customer Dataset v2", timestamp: new Date() },
                    { action: "Updated", product: "Sales Analytics", timestamp: new Date() },
                    { action: "Published", product: "Product Catalog", timestamp: new Date() }
                ],
                productsByType: {
                    dataset: 80,
                    collection: 30,
                    service: 20,
                    text: 15,
                    image: 5
                }
            });
        },

        /**
         * Validate selected products
         * @private
         */
        _validateSelectedProducts: async function (aContexts) {
            const oModel = this.getView().getModel();
            let iSuccessCount = 0;
            let iFailureCount = 0;
            
            for (const oContext of aContexts) {
                try {
                    const sPath = oContext.getPath();
                    // Call validation action
                    await oModel.bindContext(`${sPath}/validateMetadata(...)`).execute();
                    iSuccessCount++;
                } catch (error) {
                    iFailureCount++;
                }
            }
            
            MessageToast.show(`Validation complete: ${iSuccessCount} passed, ${iFailureCount} failed`);
        },

        /**
         * Export catalog in specified format
         * @private
         */
        _exportCatalog: async function (sFormat) {
            try {
                const oModel = this.getView().getModel();
                // Call export action
                const oResult = await oModel.bindContext(`/exportCatalog(...)`).setParameter("format", sFormat).execute();
                
                // Trigger download
                const sUrl = oResult.getBoundContext().getObject().downloadUrl;
                window.open(sUrl, "_blank");
                
                MessageToast.show("Export started successfully");
            } catch (error) {
                MessageToast.show("Export failed");
            }
        },

        /**
         * Format status for display
         */
        formatStatus: function (sStatus) {
            const oResourceBundle = this.getView().getModel("i18n").getResourceBundle();
            return oResourceBundle.getText(`status.${sStatus.toLowerCase()}`);
        },

        /**
         * Format quality score with color
         */
        formatQualityState: function (iScore) {
            if (iScore >= 80) return "Success";
            if (iScore >= 60) return "Warning";
            return "Error";
        }
    };
});