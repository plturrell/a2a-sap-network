sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/ui/core/Fragment",
    "sap/m/MessageBox",
    "sap/m/MessageToast"
], function (ControllerExtension, Fragment, MessageBox, MessageToast) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent1.ext.controller.ListReportExt", {
        
        override: {
            onInit: function () {
                this._extensionAPI = this.base.getExtensionAPI();
            }
        },

        onCreateStandardizationTask: function() {
            var oView = this.base.getView();
            
            if (!this._oCreateDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent1.ext.fragment.CreateStandardizationTask",
                    controller: this
                }).then(function(oDialog) {
                    this._oCreateDialog = oDialog;
                    oView.addDependent(this._oCreateDialog);
                    this._oCreateDialog.open();
                }.bind(this));
            } else {
                this._oCreateDialog.open();
            }
        },

        onImportSchema: function() {
            var oView = this.base.getView();
            
            if (!this._oImportSchemaDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent1.ext.fragment.ImportSchema",
                    controller: this
                }).then(function(oDialog) {
                    this._oImportSchemaDialog = oDialog;
                    oView.addDependent(this._oImportSchemaDialog);
                    this._oImportSchemaDialog.open();
                }.bind(this));
            } else {
                this._oImportSchemaDialog.open();
            }
        },

        onBatchProcess: function() {
            var oTable = this._extensionAPI.getTable();
            var aSelectedContexts = oTable.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageBox.warning("Please select at least one task for batch processing.");
                return;
            }
            
            var aTaskNames = aSelectedContexts.map(function(oContext) {
                return oContext.getProperty("taskName");
            });
            
            MessageBox.confirm(
                "Start batch processing for " + aSelectedContexts.length + " tasks?\n\n" +
                "Tasks: " + aTaskNames.join(", "),
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._startBatchProcessing(aSelectedContexts);
                        }
                    }.bind(this)
                }
            );
        },

        _startBatchProcessing: function(aContexts) {
            var aTaskIds = aContexts.map(function(oContext) {
                return oContext.getProperty("ID");
            });
            
            // Show busy indicator
            this.base.getView().setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent1/v1/batch-process",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    taskIds: aTaskIds,
                    parallel: true,
                    priority: "HIGH"
                }),
                success: function(data) {
                    this.base.getView().setBusy(false);
                    MessageBox.success(
                        "Batch processing started successfully!\n" +
                        "Job ID: " + data.jobId + "\n" +
                        "Processing " + data.taskCount + " tasks"
                    );
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    this.base.getView().setBusy(false);
                    MessageBox.error("Batch processing failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        onSchemaTemplates: function() {
            // Navigate to schema templates
            var oRouter = sap.ui.core.UIComponent.getRouterFor(this.base.getView());
            oRouter.navTo("SchemaTemplates");
        },

        onAnalyzeFormats: function() {
            var oView = this.base.getView();
            
            if (!this._oFormatAnalyzer) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent1.ext.fragment.FormatAnalyzer",
                    controller: this
                }).then(function(oDialog) {
                    this._oFormatAnalyzer = oDialog;
                    oView.addDependent(this._oFormatAnalyzer);
                    this._oFormatAnalyzer.open();
                    this._loadFormatStatistics();
                }.bind(this));
            } else {
                this._oFormatAnalyzer.open();
                this._loadFormatStatistics();
            }
        },

        _loadFormatStatistics: function() {
            jQuery.ajax({
                url: "/a2a/agent1/v1/format-statistics",
                type: "GET",
                success: function(data) {
                    var oModel = new sap.ui.model.json.JSONModel(data);
                    this._oFormatAnalyzer.setModel(oModel, "stats");
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to load format statistics");
                }
            });
        }
    });
});