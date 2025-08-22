sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/ui/core/Fragment",
    "sap/m/MessageBox"
], function (ControllerExtension, Fragment, MessageBox) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent0.ext.controller.ListReportExt", {
        
        override: {
            onInit: function () {
                // Access the extensionAPI via this.base.getExtensionAPI()
                this._extensionAPI = this.base.getExtensionAPI();
            }
        },

        onCreateDataProduct: function() {
            var oView = this.base.getView();
            
            // Create wizard dialog
            if (!this._oWizardDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent0.ext.fragment.DataProductWizard",
                    controller: this
                }).then(function(oDialog) {
                    this._oWizardDialog = oDialog;
                    oView.addDependent(this._oWizardDialog);
                    this._oWizardDialog.open();
                }.bind(this));
            } else {
                this._oWizardDialog.open();
            }
        },

        onBulkUpload: function() {
            MessageBox.information("Bulk upload functionality will be available soon.");
        },

        onExportData: function() {
            var oTable = this._extensionAPI.getTable();
            var aSelectedContexts = oTable.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageBox.warning("Please select at least one data product to export.");
                return;
            }
            
            // Implement export logic
            this._exportSelectedProducts(aSelectedContexts);
        },

        _exportSelectedProducts: function(aContexts) {
            var aExportData = aContexts.map(function(oContext) {
                return oContext.getObject();
            });
            
            // Convert to CSV or JSON based on user preference
            var sCSV = this._convertToCSV(aExportData);
            this._downloadFile(sCSV, "data-products-export.csv", "text/csv");
        },

        _convertToCSV: function(aData) {
            if (aData.length === 0) return "";
            
            var aKeys = Object.keys(aData[0]);
            var sHeader = aKeys.join(",");
            
            var aRows = aData.map(function(oRow) {
                return aKeys.map(function(sKey) {
                    var value = oRow[sKey];
                    // Escape quotes and wrap in quotes if contains comma
                    if (typeof value === "string" && (value.includes(",") || value.includes('"'))) {
                        value = '"' + value.replace(/"/g, '""') + '"';
                    }
                    return value || "";
                }).join(",");
            });
            
            return sHeader + "\n" + aRows.join("\n");
        },

        _downloadFile: function(sContent, sFileName, sMimeType) {
            var oBlob = new Blob([sContent], { type: sMimeType });
            var sUrl = URL.createObjectURL(oBlob);
            
            var oLink = document.createElement("a");
            oLink.href = sUrl;
            oLink.download = sFileName;
            document.body.appendChild(oLink);
            oLink.click();
            document.body.removeChild(oLink);
            URL.revokeObjectURL(sUrl);
        },

        onRefreshData: function() {
            this._extensionAPI.refresh();
            MessageBox.success("Data refreshed successfully.");
        }
    });
});