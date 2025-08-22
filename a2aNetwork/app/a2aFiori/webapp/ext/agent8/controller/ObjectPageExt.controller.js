sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel"
], function (ControllerExtension, MessageBox, MessageToast, Fragment, JSONModel) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent8.ext.controller.ObjectPageExt", {
        
        override: {
            onInit: function () {
                this._extensionAPI = this.base.getExtensionAPI();
            }
        },

        onStoreData: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sDatasetName = oContext.getProperty("datasetName");
            
            if (!this._oStoreDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent8.ext.fragment.StoreData",
                    controller: this
                }).then(function(oDialog) {
                    this._oStoreDialog = oDialog;
                    this.base.getView().addDependent(this._oStoreDialog);
                    
                    var oModel = new JSONModel({
                        datasetName: sDatasetName,
                        dataFormat: "JSON",
                        compressionType: "GZIP",
                        encryptionEnabled: false,
                        createVersion: true,
                        cacheAfterStore: true,
                        validationEnabled: true
                    });
                    this._oStoreDialog.setModel(oModel, "store");
                    this._oStoreDialog.open();
                }.bind(this));
            } else {
                this._oStoreDialog.open();
            }
        },

        onRetrieveData: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sDatasetName = oContext.getProperty("datasetName");
            var sCurrentVersion = oContext.getProperty("currentVersion");
            
            if (!this._oRetrieveDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent8.ext.fragment.RetrieveData",
                    controller: this
                }).then(function(oDialog) {
                    this._oRetrieveDialog = oDialog;
                    this.base.getView().addDependent(this._oRetrieveDialog);
                    
                    var oModel = new JSONModel({
                        datasetName: sDatasetName,
                        version: sCurrentVersion,
                        outputFormat: "JSON",
                        useCache: true,
                        filterCriteria: {},
                        maxRecords: 0,
                        includeMetadata: false
                    });
                    this._oRetrieveDialog.setModel(oModel, "retrieve");
                    this._oRetrieveDialog.open();
                    this._loadAvailableVersions(sDatasetName);
                }.bind(this));
            } else {
                this._oRetrieveDialog.open();
                this._loadAvailableVersions(sDatasetName);
            }
        },

        _loadAvailableVersions: function(sDatasetName) {
            jQuery.ajax({
                url: "/a2a/agent8/v1/datasets/" + sDatasetName + "/versions",
                type: "GET",
                success: function(data) {
                    var oModel = this._oRetrieveDialog.getModel("retrieve");
                    var oData = oModel.getData();
                    oData.availableVersions = data.versions;
                    oModel.setData(oData);
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to load versions: " + xhr.responseText);
                }.bind(this)
            });
        },

        onCacheData: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sDatasetName = oContext.getProperty("datasetName");
            
            MessageBox.confirm("Cache dataset '" + sDatasetName + "' in memory?", {
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        this._executeCacheOperation(sDatasetName);
                    }
                }.bind(this)
            });
        },

        _executeCacheOperation: function(sDatasetName) {
            this._extensionAPI.getView().setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent8/v1/datasets/" + sDatasetName + "/cache",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    cacheLevel: "MEMORY",
                    ttl: 3600,
                    preload: true
                }),
                success: function(data) {
                    this._extensionAPI.getView().setBusy(false);
                    MessageBox.success(
                        "Dataset cached successfully!\\n" +
                        "Cache size: " + data.cacheSize + "\\n" +
                        "Records cached: " + data.recordCount + "\\n" +
                        "Cache key: " + data.cacheKey
                    );
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    this._extensionAPI.getView().setBusy(false);
                    MessageBox.error("Cache operation failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        onCreateVersion: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sDatasetName = oContext.getProperty("datasetName");
            
            if (!this._oVersionDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent8.ext.fragment.CreateVersion",
                    controller: this
                }).then(function(oDialog) {
                    this._oVersionDialog = oDialog;
                    this.base.getView().addDependent(this._oVersionDialog);
                    
                    var oModel = new JSONModel({
                        datasetName: sDatasetName,
                        versionType: "INCREMENTAL",
                        versionComment: "",
                        createBackup: true,
                        validateIntegrity: true,
                        compressVersion: true
                    });
                    this._oVersionDialog.setModel(oModel, "version");
                    this._oVersionDialog.open();
                }.bind(this));
            } else {
                this._oVersionDialog.open();
            }
        },

        onOptimizeStorage: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sDatasetName = oContext.getProperty("datasetName");
            
            MessageBox.confirm(
                "Optimize storage for dataset '" + sDatasetName + "'? This may take several minutes.",
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._optimizeStorage(sDatasetName);
                        }
                    }.bind(this)
                }
            );
        },

        _optimizeStorage: function(sDatasetName) {
            this._extensionAPI.getView().setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent8/v1/datasets/" + sDatasetName + "/optimize",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    compressionLevel: "OPTIMAL",
                    defragment: true,
                    rebuildIndexes: true,
                    updateStatistics: true
                }),
                success: function(data) {
                    this._extensionAPI.getView().setBusy(false);
                    
                    MessageBox.success(
                        "Storage optimization completed!\\n" +
                        "Space saved: " + data.spaceSaved + "\\n" +
                        "Compression improved: " + data.compressionImprovement + "%\\n" +
                        "Optimization time: " + data.optimizationTime + "s"
                    );
                    
                    this._extensionAPI.refresh();
                    this._startOptimizationMonitoring(data.optimizationId);
                }.bind(this),
                error: function(xhr) {
                    this._extensionAPI.getView().setBusy(false);
                    MessageBox.error("Storage optimization failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        _startOptimizationMonitoring: function(sOptimizationId) {
            this._optimizationEventSource = new EventSource("/a2a/agent8/v1/optimizations/" + sOptimizationId + "/stream");
            
            this._optimizationEventSource.onmessage = function(event) {
                var data = JSON.parse(event.data);
                
                if (data.type === "progress") {
                    MessageToast.show("Optimization: " + data.stage + " (" + data.progress + "%)");
                } else if (data.type === "complete") {
                    this._optimizationEventSource.close();
                    MessageToast.show("Storage optimization completed successfully");
                } else if (data.type === "error") {
                    this._optimizationEventSource.close();
                    MessageBox.error("Optimization failed: " + data.error);
                }
            }.bind(this);
            
            this._optimizationEventSource.onerror = function() {
                this._optimizationEventSource.close();
            }.bind(this);
        },

        onClearCache: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sDatasetName = oContext.getProperty("datasetName");
            
            MessageBox.confirm("Clear all cache entries for '" + sDatasetName + "'?", {
                onClose: function(oAction) {
                    if (oAction === MessageBox.Action.OK) {
                        this._clearCache(sDatasetName);
                    }
                }.bind(this)
            });
        },

        _clearCache: function(sDatasetName) {
            jQuery.ajax({
                url: "/a2a/agent8/v1/datasets/" + sDatasetName + "/cache",
                type: "DELETE",
                success: function(data) {
                    MessageToast.show(
                        "Cache cleared: " + data.entriesRemoved + " entries removed, " +
                        data.memoryFreed + " memory freed"
                    );
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to clear cache: " + xhr.responseText);
                }.bind(this)
            });
        },

        onValidateData: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sDatasetName = oContext.getProperty("datasetName");
            
            this._extensionAPI.getView().setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent8/v1/datasets/" + sDatasetName + "/validate",
                type: "POST",
                success: function(data) {
                    this._extensionAPI.getView().setBusy(false);
                    this._showValidationResults(data);
                }.bind(this),
                error: function(xhr) {
                    this._extensionAPI.getView().setBusy(false);
                    MessageBox.error("Data validation failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        _showValidationResults: function(validationData) {
            var sMessage = "Data Validation Results:\\n\\n";
            
            sMessage += "Total Records: " + validationData.totalRecords + "\\n";
            sMessage += "Valid Records: " + validationData.validRecords + "\\n";
            sMessage += "Invalid Records: " + validationData.invalidRecords + "\\n";
            sMessage += "Validation Score: " + validationData.validationScore + "%\\n\\n";
            
            if (validationData.errors && validationData.errors.length > 0) {
                sMessage += "Validation Errors:\\n";
                validationData.errors.slice(0, 5).forEach(function(error) {
                    sMessage += "• " + error.field + ": " + error.message + "\\n";
                });
                
                if (validationData.errors.length > 5) {
                    sMessage += "... and " + (validationData.errors.length - 5) + " more errors\\n";
                }
            }
            
            MessageBox.information(sMessage, {
                actions: ["Export Report", MessageBox.Action.CLOSE],
                onClose: function(oAction) {
                    if (oAction === "Export Report") {
                        this._exportValidationReport(validationData);
                    }
                }.bind(this)
            });
        },

        _exportValidationReport: function(validationData) {
            var sDatasetName = this._extensionAPI.getBindingContext().getProperty("datasetName");
            
            jQuery.ajax({
                url: "/a2a/agent8/v1/datasets/" + sDatasetName + "/validation-report",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify({
                    validationData: validationData,
                    format: "PDF"
                }),
                success: function(data) {
                    MessageToast.show("Validation report exported successfully");
                    window.open(data.downloadUrl, "_blank");
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to export report: " + xhr.responseText);
                }.bind(this)
            });
        },

        onCompressData: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sDatasetName = oContext.getProperty("datasetName");
            
            if (!this._oCompressDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent8.ext.fragment.CompressData",
                    controller: this
                }).then(function(oDialog) {
                    this._oCompressDialog = oDialog;
                    this.base.getView().addDependent(this._oCompressDialog);
                    
                    var oModel = new JSONModel({
                        datasetName: sDatasetName,
                        compressionType: "GZIP",
                        compressionLevel: "STANDARD",
                        preserveOriginal: true,
                        validateAfterCompression: true
                    });
                    this._oCompressDialog.setModel(oModel, "compress");
                    this._oCompressDialog.open();
                }.bind(this));
            } else {
                this._oCompressDialog.open();
            }
        },

        onBackupData: function() {
            var oContext = this._extensionAPI.getBindingContext();
            var sDatasetName = oContext.getProperty("datasetName");
            
            if (!this._oBackupDialog) {
                Fragment.load({
                    id: this.base.getView().getId(),
                    name: "a2a.network.agent8.ext.fragment.BackupData",
                    controller: this
                }).then(function(oDialog) {
                    this._oBackupDialog = oDialog;
                    this.base.getView().addDependent(this._oBackupDialog);
                    
                    var oModel = new JSONModel({
                        datasetName: sDatasetName,
                        backupType: "FULL",
                        compressionEnabled: true,
                        encryptionEnabled: true,
                        backupLocation: "S3",
                        retentionDays: 90
                    });
                    this._oBackupDialog.setModel(oModel, "backup");
                    this._oBackupDialog.open();
                }.bind(this));
            } else {
                this._oBackupDialog.open();
            }
        },

        onExecuteStoreData: function() {
            var oModel = this._oStoreDialog.getModel("store");
            var oData = oModel.getData();
            
            this._oStoreDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent8/v1/datasets/" + oData.datasetName + "/store",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify(oData),
                success: function(data) {
                    this._oStoreDialog.setBusy(false);
                    this._oStoreDialog.close();
                    
                    MessageBox.success(
                        "Data stored successfully!\\n" +
                        "Records stored: " + data.recordsStored + "\\n" +
                        "Storage size: " + data.storageSize + "\\n" +
                        "Compression ratio: " + data.compressionRatio + "%"
                    );
                    
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    this._oStoreDialog.setBusy(false);
                    MessageBox.error("Store operation failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        onExecuteRetrieveData: function() {
            var oModel = this._oRetrieveDialog.getModel("retrieve");
            var oData = oModel.getData();
            
            this._oRetrieveDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent8/v1/datasets/" + oData.datasetName + "/retrieve",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify(oData),
                success: function(data) {
                    this._oRetrieveDialog.setBusy(false);
                    this._oRetrieveDialog.close();
                    
                    this._showRetrievedData(data);
                }.bind(this),
                error: function(xhr) {
                    this._oRetrieveDialog.setBusy(false);
                    MessageBox.error("Retrieve operation failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        _showRetrievedData: function(retrievedData) {
            var oView = this.base.getView();
            
            if (!this._oDataViewerDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent8.ext.fragment.DataViewer",
                    controller: this
                }).then(function(oDialog) {
                    this._oDataViewerDialog = oDialog;
                    oView.addDependent(this._oDataViewerDialog);
                    
                    var oModel = new JSONModel(retrievedData);
                    this._oDataViewerDialog.setModel(oModel, "data");
                    this._oDataViewerDialog.open();
                }.bind(this));
            } else {
                var oModel = new JSONModel(retrievedData);
                this._oDataViewerDialog.setModel(oModel, "data");
                this._oDataViewerDialog.open();
            }
        },

        onConfirmCreateVersion: function() {
            var oModel = this._oVersionDialog.getModel("version");
            var oData = oModel.getData();
            
            this._oVersionDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent8/v1/datasets/" + oData.datasetName + "/versions",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify(oData),
                success: function(data) {
                    this._oVersionDialog.setBusy(false);
                    this._oVersionDialog.close();
                    
                    MessageBox.success(
                        "Version created successfully!\\n" +
                        "Version ID: " + data.versionId + "\\n" +
                        "Version Number: " + data.versionNumber + "\\n" +
                        "Size: " + data.versionSize
                    );
                    
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    this._oVersionDialog.setBusy(false);
                    MessageBox.error("Version creation failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        onConfirmCompress: function() {
            var oModel = this._oCompressDialog.getModel("compress");
            var oData = oModel.getData();
            
            this._oCompressDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent8/v1/datasets/" + oData.datasetName + "/compress",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify(oData),
                success: function(data) {
                    this._oCompressDialog.setBusy(false);
                    this._oCompressDialog.close();
                    
                    MessageBox.success(
                        "Data compression completed!\\n" +
                        "Original size: " + data.originalSize + "\\n" +
                        "Compressed size: " + data.compressedSize + "\\n" +
                        "Compression ratio: " + data.compressionRatio + "%"
                    );
                    
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    this._oCompressDialog.setBusy(false);
                    MessageBox.error("Compression failed: " + xhr.responseText);
                }.bind(this)
            });
        },

        onConfirmBackup: function() {
            var oModel = this._oBackupDialog.getModel("backup");
            var oData = oModel.getData();
            
            this._oBackupDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent8/v1/datasets/" + oData.datasetName + "/backup",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify(oData),
                success: function(data) {
                    this._oBackupDialog.setBusy(false);
                    this._oBackupDialog.close();
                    
                    MessageBox.success(
                        "Backup created successfully!\\n" +
                        "Backup ID: " + data.backupId + "\\n" +
                        "Backup size: " + data.backupSize + "\\n" +
                        "Location: " + data.backupLocation
                    );
                    
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    this._oBackupDialog.setBusy(false);
                    MessageBox.error("Backup failed: " + xhr.responseText);
                }.bind(this)
            });
        }
    });
});