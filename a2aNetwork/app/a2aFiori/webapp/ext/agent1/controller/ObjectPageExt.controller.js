sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/core/Fragment",
    "sap/base/security/encodeXML",
    "sap/base/strings/escapeRegExp"
], function (ControllerExtension, MessageBox, MessageToast, Fragment, encodeXML, escapeRegExp) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent1.ext.controller.ObjectPageExt", {
        
        override: {
            onInit: function () {
                this._extensionAPI = this.base.getExtensionAPI();
            }
        },

        onStartStandardization: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            var sTaskName = oContext.getProperty("taskName");
            
            MessageBox.confirm("Start standardization for '" + sTaskName + "'?", {
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        this._startStandardizationProcess(sTaskId);
                    }
                }.bind(this)
            });
        },

        _startStandardizationProcess: function(sTaskId) {
            this._extensionAPI.getView().setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent1/v1/tasks/" + sTaskId + "/start",
                type: "POST",
                success: function(data) {
                    this._extensionAPI.getView().setBusy(false);
                    MessageToast.show("Standardization process started");
                    this._extensionAPI.refresh();
                    
                    // Start monitoring progress
                    this._startProgressMonitoring(sTaskId);
                }.bind(this),
                error: function(xhr) {
                    this._extensionAPI.getView().setBusy(false);
                    MessageBox.error("Failed to start standardization: " + xhr.responseText);
                }.bind(this)
            });
        },

        _startProgressMonitoring: function(sTaskId) {
            // Poll for progress updates every 2 seconds
            this._progressInterval = setInterval(function() {
                jQuery.ajax({
                    url: "/a2a/agent1/v1/tasks/" + sTaskId + "/progress",
                    type: "GET",
                    success: function(data) {
                        if (data.status === "COMPLETED" || data.status === "FAILED") {
                            clearInterval(this._progressInterval);
                            this._extensionAPI.refresh();
                            
                            if (data.status === "COMPLETED") {
                                MessageBox.success("Standardization completed successfully!");
                            } else {
                                MessageBox.error("Standardization failed: " + data.error);
                            }
                        }
                    }.bind(this),
                    error: function() {
                        clearInterval(this._progressInterval);
                    }.bind(this)
                });
            }.bind(this), 2000);
        },

        onPauseStandardization: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            
            MessageBox.confirm("Pause standardization process?", {
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        this._pauseStandardization(sTaskId);
                    }
                }.bind(this)
            });
        },

        _pauseStandardization: function(sTaskId) {
            jQuery.ajax({
                url: "/a2a/agent1/v1/tasks/" + sTaskId + "/pause",
                type: "POST",
                success: function() {
                    MessageToast.show("Standardization process paused");
                    clearInterval(this._progressInterval);
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to pause standardization: " + xhr.responseText);
                }
            });
        },

        onValidateMapping: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var oData = oContext.getObject();
            
            if (!oData.mappingRules || oData.mappingRules.length === 0) {
                MessageBox.warning("No mapping rules defined. Please define mapping rules first.");
                return;
            }
            
            this._extensionAPI.getView().setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent1/v1/validate-mapping",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    sourceSchema: oData.sourceSchema,
                    targetSchema: oData.targetSchema,
                    mappingRules: oData.mappingRules
                }),
                success: function(data) {
                    this._extensionAPI.getView().setBusy(false);
                    
                    if (data.isValid) {
                        MessageBox.success(
                            "Schema mapping is valid!\n\n" +
                            "Coverage: " + data.coverage + "%\n" +
                            "Mapped fields: " + data.mappedFields + "/" + data.totalFields
                        );
                    } else {
                        this._showValidationErrors(data.errors);
                    }
                }.bind(this),
                error: function(xhr) {
                    this._extensionAPI.getView().setBusy(false);
                    MessageBox.error("Validation failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        _showValidationErrors: function(aErrors) {
            var oView = this.base.getView();
            
            if (!this._oValidationErrorsDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent1.ext.fragment.ValidationErrors",
                    controller: this
                }).then(function(oDialog) {
                    this._oValidationErrorsDialog = oDialog;
                    oView.addDependent(this._oValidationErrorsDialog);
                    
                    var oModel = new sap.ui.model.json.JSONModel({ errors: aErrors });
                    this._oValidationErrorsDialog.setModel(oModel, "errors");
                    this._oValidationErrorsDialog.open();
                }.bind(this));
            } else {
                var oModel = new sap.ui.model.json.JSONModel({ errors: aErrors });
                this._oValidationErrorsDialog.setModel(oModel, "errors");
                this._oValidationErrorsDialog.open();
            }
        },

        onExportResults: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sTaskId = oContext.getProperty("ID");
            var sTaskName = oContext.getProperty("taskName");
            
            var oExportOptions = {
                formats: ["CSV", "JSON", "PARQUET"],
                includeErrors: true,
                includeMetadata: true
            };
            
            this._showExportDialog(sTaskId, sTaskName, oExportOptions);
        },

        _showExportDialog: function(sTaskId, sTaskName, oOptions) {
            var oView = this.base.getView();
            
            if (!this._oExportDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent1.ext.fragment.ExportResults",
                    controller: this
                }).then(function(oDialog) {
                    this._oExportDialog = oDialog;
                    oView.addDependent(this._oExportDialog);
                    
                    var oModel = new sap.ui.model.json.JSONModel({
                        taskId: sTaskId,
                        taskName: sTaskName,
                        options: oOptions,
                        selectedFormat: "CSV"
                    });
                    this._oExportDialog.setModel(oModel, "export");
                    this._oExportDialog.open();
                }.bind(this));
            } else {
                var oModel = new sap.ui.model.json.JSONModel({
                    taskId: sTaskId,
                    taskName: sTaskName,
                    options: oOptions,
                    selectedFormat: "CSV"
                });
                this._oExportDialog.setModel(oModel, "export");
                this._oExportDialog.open();
            }
        },

        onExecuteExport: function() {
            var oExportModel = this._oExportDialog.getModel("export");
            var oExportData = oExportModel.getData();
            
            this._oExportDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent1/v1/tasks/" + oExportData.taskId + "/export",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    format: oExportData.selectedFormat,
                    includeErrors: oExportData.options.includeErrors,
                    includeMetadata: oExportData.options.includeMetadata
                }),
                success: function(data) {
                    this._oExportDialog.setBusy(false);
                    this._oExportDialog.close();
                    
                    // Download the exported file
                    window.open(data.downloadUrl, "_blank");
                    MessageToast.show("Export completed successfully");
                }.bind(this),
                error: function(xhr) {
                    this._oExportDialog.setBusy(false);
                    MessageBox.error("Export failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        onPreviewTransformation: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var oData = oContext.getObject();
            
            if (!oData.mappingRules || oData.mappingRules.length === 0) {
                MessageBox.warning("No mapping rules defined for preview.");
                return;
            }
            
            // Open preview dialog with sample data transformation
            this._showTransformationPreview(oData);
        },

        _showTransformationPreview: function(oTaskData) {
            var oView = this.base.getView();
            
            if (!this._oPreviewDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent1.ext.fragment.TransformationPreview",
                    controller: this
                }).then(function(oDialog) {
                    this._oPreviewDialog = oDialog;
                    oView.addDependent(this._oPreviewDialog);
                    
                    this._loadPreviewData(oTaskData);
                    this._oPreviewDialog.open();
                }.bind(this));
            } else {
                this._loadPreviewData(oTaskData);
                this._oPreviewDialog.open();
            }
        },

        _loadPreviewData: function(oTaskData) {
            jQuery.ajax({
                url: "/a2a/agent1/v1/preview-transformation",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    sourceSchema: oTaskData.sourceSchema,
                    targetSchema: oTaskData.targetSchema,
                    mappingRules: oTaskData.mappingRules,
                    sampleSize: 5
                }),
                success: function(data) {
                    var oModel = new sap.ui.model.json.JSONModel(data);
                    this._oPreviewDialog.setModel(oModel, "preview");
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to generate preview: " + xhr.responseText);
                }
            });
        }
    });
});