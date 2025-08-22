sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageBox",
    "sap/m/MessageToast"
], function (ControllerExtension, MessageBox, MessageToast) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent0.ext.controller.ObjectPageExt", {
        
        override: {
            onInit: function () {
                this._extensionAPI = this.base.getExtensionAPI();
            }
        },

        onValidateQuality: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sProductId = oContext.getProperty("ID");
            var sProductName = oContext.getProperty("name");
            
            MessageBox.confirm("Validate quality for '" + sProductName + "'?", {
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        this._performQualityValidation(sProductId);
                    }
                }.bind(this)
            });
        },

        _performQualityValidation: function(sProductId) {
            // Show busy indicator
            this._extensionAPI.getView().setBusy(true);
            
            // Call Agent 0 validation endpoint
            jQuery.ajax({
                url: "/a2a/agent0/v1/data-products/" + sProductId + "/validate",
                type: "POST",
                success: function(data) {
                    this._extensionAPI.getView().setBusy(false);
                    
                    if (data.qualityScore >= 80) {
                        MessageBox.success(
                            "Quality validation passed!\n" +
                            "Score: " + data.qualityScore + "/100\n" +
                            "Status: " + data.validationStatus
                        );
                    } else {
                        MessageBox.warning(
                            "Quality validation completed with issues.\n" +
                            "Score: " + data.qualityScore + "/100\n" +
                            "Issues: " + data.validationErrors.join(", ")
                        );
                    }
                    
                    // Refresh the binding to show updated data
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    this._extensionAPI.getView().setBusy(false);
                    MessageBox.error("Validation failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        onRegisterORD: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sProductId = oContext.getProperty("ID");
            var sProductName = oContext.getProperty("name");
            
            MessageBox.confirm("Register '" + sProductName + "' in ORD Registry?", {
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        this._registerInORD(sProductId);
                    }
                }.bind(this)
            });
        },

        _registerInORD: function(sProductId) {
            this._extensionAPI.getView().setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent0/v1/data-products/" + sProductId + "/register-ord",
                type: "POST",
                success: function(data) {
                    this._extensionAPI.getView().setBusy(false);
                    
                    MessageBox.success(
                        "Successfully registered in ORD!\n" +
                        "ORD ID: " + data.ordRegistryId + "\n" +
                        "Package: " + data.packageId
                    );
                    
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    this._extensionAPI.getView().setBusy(false);
                    MessageBox.error("ORD registration failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        onDownloadMetadata: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var oData = oContext.getObject();
            
            // Extract Dublin Core metadata
            var oMetadata = {
                dublinCore: {
                    title: oData.dcTitle,
                    creator: oData.dcCreator,
                    subject: oData.dcSubject,
                    description: oData.dcDescription,
                    publisher: oData.dcPublisher,
                    contributor: oData.dcContributor,
                    date: oData.dcDate,
                    type: oData.dcType,
                    format: oData.dcFormat,
                    identifier: oData.dcIdentifier,
                    source: oData.dcSource,
                    language: oData.dcLanguage,
                    relation: oData.dcRelation,
                    coverage: oData.dcCoverage,
                    rights: oData.dcRights
                },
                technical: {
                    format: oData.format,
                    fileSize: oData.fileSize,
                    integrityHash: oData.integrityHash,
                    qualityScore: oData.qualityScore
                }
            };
            
            var sJSON = JSON.stringify(oMetadata, null, 2);
            this._downloadFile(sJSON, oData.name + "-metadata.json", "application/json");
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

        onViewLineage: function() {
            MessageToast.show("Data lineage visualization coming soon...");
        }
    });
});