sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/model/json/JSONModel"
], function(Controller, MessageBox, MessageToast, JSONModel) {
    "use strict";

    return Controller.extend("a2a.network.fiori.ext.agent0.DataProductCreationWizard", {
        
        onInit: function() {
            // Initialize wizard model
            var oWizardModel = new JSONModel({
                dataProduct: {
                    name: "",
                    description: "",
                    format: "",
                    file: null,
                    metadata: {
                        dcTitle: "",
                        dcCreator: "",
                        dcSubject: "",
                        dcDescription: "",
                        dcPublisher: "",
                        dcContributor: "",
                        dcDate: new Date(),
                        dcType: "",
                        dcFormat: "",
                        dcIdentifier: "",
                        dcSource: "",
                        dcLanguage: "en",
                        dcRelation: "",
                        dcCoverage: "",
                        dcRights: ""
                    },
                    transformationRules: [],
                    qualityThreshold: 80
                }
            });
            this.getView().setModel(oWizardModel, "wizard");
        },

        onFileChange: function(oEvent) {
            var aFiles = oEvent.getParameter("files");
            if (aFiles && aFiles.length > 0) {
                var oFile = aFiles[0];
                var oWizardModel = this.getView().getModel("wizard");
                
                // Update model with file info
                oWizardModel.setProperty("/dataProduct/file", oFile);
                oWizardModel.setProperty("/dataProduct/format", this._detectFormat(oFile.name));
                oWizardModel.setProperty("/dataProduct/metadata/dcFormat", oFile.type);
                
                // Auto-populate some metadata
                oWizardModel.setProperty("/dataProduct/metadata/dcTitle", oFile.name);
                oWizardModel.setProperty("/dataProduct/metadata/dcDate", new Date());
                
                MessageToast.show("File uploaded: " + oFile.name);
            }
        },

        _detectFormat: function(sFileName) {
            var sExtension = sFileName.split('.').pop().toLowerCase();
            var formatMap = {
                "csv": "CSV",
                "json": "JSON",
                "xml": "XML",
                "parquet": "PARQUET",
                "txt": "TEXT",
                "xlsx": "EXCEL",
                "xls": "EXCEL"
            };
            return formatMap[sExtension] || "UNKNOWN";
        },

        onValidateStep1: function() {
            var oWizardModel = this.getView().getModel("wizard");
            var oData = oWizardModel.getProperty("/dataProduct");
            
            if (!oData.name || !oData.description || !oData.file) {
                MessageBox.error("Please fill all required fields and upload a file");
                return false;
            }
            return true;
        },

        onValidateMetadata: function() {
            var oWizardModel = this.getView().getModel("wizard");
            var oMetadata = oWizardModel.getProperty("/dataProduct/metadata");
            
            // Validate required Dublin Core fields
            if (!oMetadata.dcTitle || !oMetadata.dcCreator || !oMetadata.dcType) {
                MessageBox.error("Please fill all required metadata fields");
                return false;
            }
            return true;
        },

        onAddTransformationRule: function() {
            var oWizardModel = this.getView().getModel("wizard");
            var aRules = oWizardModel.getProperty("/dataProduct/transformationRules");
            
            aRules.push({
                type: "",
                field: "",
                operation: "",
                parameters: {}
            });
            
            oWizardModel.setProperty("/dataProduct/transformationRules", aRules);
        },

        onWizardComplete: function() {
            var oWizardModel = this.getView().getModel("wizard");
            var oDataProduct = oWizardModel.getProperty("/dataProduct");
            
            // Prepare FormData for file upload
            var oFormData = new FormData();
            oFormData.append("file", oDataProduct.file);
            oFormData.append("metadata", JSON.stringify({
                name: oDataProduct.name,
                description: oDataProduct.description,
                format: oDataProduct.format,
                metadata: oDataProduct.metadata,
                transformationRules: oDataProduct.transformationRules,
                qualityThreshold: oDataProduct.qualityThreshold
            }));
            
            // Call Agent 0 API
            jQuery.ajax({
                url: "/a2a/agent0/v1/data-products",
                type: "POST",
                data: oFormData,
                processData: false,
                contentType: false,
                success: function(data) {
                    MessageBox.success("Data Product created successfully!", {
                        onClose: function() {
                            this.getView().getParent().close();
                            this._resetWizard();
                        }.bind(this)
                    });
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to create Data Product: " + xhr.responseText);
                }
            });
        },

        onWizardCancel: function() {
            MessageBox.confirm("Are you sure you want to cancel?", {
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        this.getView().getParent().close();
                        this._resetWizard();
                    }
                }.bind(this)
            });
        },

        _resetWizard: function() {
            var oWizard = this.byId("dataProductWizard");
            var oWizardModel = this.getView().getModel("wizard");
            
            // Reset wizard to first step
            oWizard.discardProgress(oWizard.getSteps()[0]);
            
            // Reset model
            oWizardModel.setProperty("/dataProduct", {
                name: "",
                description: "",
                format: "",
                file: null,
                metadata: {
                    dcTitle: "",
                    dcCreator: "",
                    dcSubject: "",
                    dcDescription: "",
                    dcPublisher: "",
                    dcContributor: "",
                    dcDate: new Date(),
                    dcType: "",
                    dcFormat: "",
                    dcIdentifier: "",
                    dcSource: "",
                    dcLanguage: "en",
                    dcRelation: "",
                    dcCoverage: "",
                    dcRights: ""
                },
                transformationRules: [],
                qualityThreshold: 80
            });
        }
    });
});