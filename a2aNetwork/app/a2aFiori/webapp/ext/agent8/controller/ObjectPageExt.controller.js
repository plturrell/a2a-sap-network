sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel",
    "sap/base/security/encodeXML",
    "sap/base/security/encodeURL",
    "sap/base/strings/escapeRegExp",
    "sap/base/security/sanitizeHTML",
    "a2a/network/agent8/ext/utils/SecurityUtils"
], function (ControllerExtension, MessageBox, MessageToast, Fragment, JSONModel, encodeXML, encodeURL, escapeRegExp, sanitizeHTML, SecurityUtils) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent8.ext.controller.ObjectPageExt", {
        
        override: {
            onInit: function () {
                this._extensionAPI = this.base.getExtensionAPI();
                this._securityUtils = SecurityUtils;
                
                // Initialize device model for responsive behavior
                var oDeviceModel = new JSONModel(sap.ui.Device);
                oDeviceModel.setDefaultBindingMode(sap.ui.model.BindingMode.OneWay);
                this.base.getView().setModel(oDeviceModel, "device");
                
                // Initialize create dialog model
                this._initializeCreateModel();
            },
            
            onExit: function() {
                this._cleanupResources();
                if (this.base.onExit) {
                    this.base.onExit.apply(this, arguments);
                }
            }
        },
        
        _initializeCreateModel: function() {
            this._oCreateModel = new JSONModel({
                taskName: "",
                description: "",
                datasetName: "",
                operationType: "",
                priority: "MEDIUM",
                storageType: "HANA",
                storageBackend: "",
                compressionEnabled: true,
                compressionType: "GZIP",
                encryptionEnabled: true,
                encryptionType: "AES256",
                partitionStrategy: "NONE",
                cacheEnabled: true,
                autoWarmCache: false,
                cacheOnWrite: true,
                cacheLevel: "MEMORY",
                cacheTTL: 60,
                evictionPolicy: "LRU",
                memoryCacheSize: 2,
                redisCacheSize: 8,
                versioningEnabled: true,
                autoVersioning: false,
                incrementalBackup: true,
                versionStrategy: "INCREMENTAL",
                retentionDays: 90,
                maxVersions: 10,
                checksumValidation: true,
                checksumAlgorithm: "SHA256",
                batchSize: 1000,
                parallelProcessing: true,
                threadPoolSize: 4,
                connectionPoolSize: 10,
                connectionTimeout: 30,
                autoOptimization: true,
                indexOptimization: true,
                queryOptimization: true,
                memoryOptimization: true,
                performanceMonitoring: true,
                trackThroughput: true,
                trackResponseTimes: true,
                trackResourceUsage: true,
                availableBackends: [],
                // Validation states
                taskNameState: "None",
                datasetNameState: "None",
                operationTypeState: "None",
                taskNameStateText: "",
                datasetNameStateText: "",
                operationTypeStateText: ""
            });
        },
        
        _cleanupResources: function() {
            if (this._optimizationEventSource) {
                this._optimizationEventSource.close();
                this._optimizationEventSource = null;
            }
            if (this._oCreateDialog) {
                this._oCreateDialog.destroy();
                this._oCreateDialog = null;
            }
            if (this._oStoreDialog) {
                this._oStoreDialog.destroy();
                this._oStoreDialog = null;
            }
            if (this._oRetrieveDialog) {
                this._oRetrieveDialog.destroy();
                this._oRetrieveDialog = null;
            }
            if (this._oVersionDialog) {
                this._oVersionDialog.destroy();
                this._oVersionDialog = null;
            }
            if (this._oCompressDialog) {
                this._oCompressDialog.destroy();
                this._oCompressDialog = null;
            }
            if (this._oBackupDialog) {
                this._oBackupDialog.destroy();
                this._oBackupDialog = null;
            }
            if (this._oDataViewerDialog) {
                this._oDataViewerDialog.destroy();
                this._oDataViewerDialog = null;
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
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    MessageBox.error("Failed to load versions: " + errorMsg);
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
            
            const ajaxConfig = this._securityUtils.createSecureAjaxConfig({
                url: "/a2a/agent8/v1/datasets/" + encodeURIComponent(sDatasetName) + "/cache",
                type: "POST",
                data: JSON.stringify({
                    cacheLevel: "MEMORY",
                    ttl: 3600,
                    preload: true
                }),
                success: function(data) {
                    this._extensionAPI.getView().setBusy(false);
                    const safeCacheSize = this._securityUtils.encodeHTML(data.cacheSize || '0');
                    const safeRecordCount = parseInt(data.recordCount) || 0;
                    const safeCacheKey = this._securityUtils.encodeHTML(data.cacheKey || 'N/A');
                    MessageBox.success(
                        "Dataset cached successfully!\\n" +
                        "Cache size: " + safeCacheSize + "\\n" +
                        "Records cached: " + safeRecordCount + "\\n" +
                        "Cache key: " + safeCacheKey
                    );
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    this._extensionAPI.getView().setBusy(false);
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    MessageBox.error("Cache operation failed: " + errorMsg);
                }.bind(this)
            });
            
            jQuery.ajax(ajaxConfig);
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
                    
                    const safeSpaceSaved = this._securityUtils.encodeHTML(data.spaceSaved || '0');
                    const safeCompression = parseFloat(data.compressionImprovement) || 0;
                    const safeTime = parseFloat(data.optimizationTime) || 0;
                    MessageBox.success(
                        "Storage optimization completed!\\n" +
                        "Space saved: " + safeSpaceSaved + "\\n" +
                        "Compression improved: " + safeCompression + "%\\n" +
                        "Optimization time: " + safeTime + "s"
                    );
                    
                    this._extensionAPI.refresh();
                    this._startOptimizationMonitoring(data.optimizationId);
                }.bind(this),
                error: function(xhr) {
                    this._extensionAPI.getView().setBusy(false);
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    MessageBox.error("Storage optimization failed: " + errorMsg);
                }.bind(this)
            });
        },

        _startOptimizationMonitoring: function(sOptimizationId) {
            this._optimizationEventSource = new EventSource("/a2a/agent8/v1/optimizations/" + sOptimizationId + "/stream");
            
            this._optimizationEventSource.onmessage = function(event) {
                var data = JSON.parse(event.data);
                
                if (data.type === "progress") {
                    const safeStage = this._securityUtils.encodeHTML(data.stage || 'Unknown stage');
                    const safeProgress = parseInt(data.progress) || 0;
                    MessageToast.show("Optimization: " + safeStage + " (" + safeProgress + "%)");
                } else if (data.type === "complete") {
                    this._optimizationEventSource.close();
                    MessageToast.show("Storage optimization completed successfully");
                } else if (data.type === "error") {
                    this._optimizationEventSource.close();
                    const safeError = this._securityUtils.sanitizeErrorMessage(data.error);
                    MessageBox.error("Optimization failed: " + safeError);
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
                    const safeEntries = parseInt(data.entriesRemoved) || 0;
                    const safeMemory = this._securityUtils.encodeHTML(data.memoryFreed || '0');
                    MessageToast.show(
                        "Cache cleared: " + safeEntries + " entries removed, " +
                        safeMemory + " memory freed"
                    );
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    MessageBox.error("Failed to clear cache: " + errorMsg);
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
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    MessageBox.error("Data validation failed: " + errorMsg);
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
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    MessageBox.error("Failed to export report: " + errorMsg);
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
                    
                    const safeRecordsStored = parseInt(data.recordsStored) || 0;
                    const safeStorageSize = this._securityUtils.encodeHTML(data.storageSize || '0');
                    const safeCompressionRatio = parseFloat(data.compressionRatio) || 0;
                    MessageBox.success(
                        "Data stored successfully!\\n" +
                        "Records stored: " + safeRecordsStored + "\\n" +
                        "Storage size: " + safeStorageSize + "\\n" +
                        "Compression ratio: " + safeCompressionRatio + "%"
                    );
                    
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    this._oStoreDialog.setBusy(false);
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    MessageBox.error("Store operation failed: " + errorMsg);
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
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    MessageBox.error("Retrieve operation failed: " + errorMsg);
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
                    
                    const safeVersionId = this._securityUtils.encodeHTML(data.versionId || 'N/A');
                    const safeVersionNumber = this._securityUtils.encodeHTML(data.versionNumber || 'N/A');
                    const safeVersionSize = this._securityUtils.encodeHTML(data.versionSize || '0');
                    MessageBox.success(
                        "Version created successfully!\\n" +
                        "Version ID: " + safeVersionId + "\\n" +
                        "Version Number: " + safeVersionNumber + "\\n" +
                        "Size: " + safeVersionSize
                    );
                    
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    this._oVersionDialog.setBusy(false);
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    MessageBox.error("Version creation failed: " + errorMsg);
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
                    
                    const safeOriginalSize = this._securityUtils.encodeHTML(data.originalSize || '0');
                    const safeCompressedSize = this._securityUtils.encodeHTML(data.compressedSize || '0');
                    const safeRatio = parseFloat(data.compressionRatio) || 0;
                    MessageBox.success(
                        "Data compression completed!\\n" +
                        "Original size: " + safeOriginalSize + "\\n" +
                        "Compressed size: " + safeCompressedSize + "\\n" +
                        "Compression ratio: " + safeRatio + "%"
                    );
                    
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    this._oCompressDialog.setBusy(false);
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    MessageBox.error("Compression failed: " + errorMsg);
                }.bind(this)
            });
        },

        onConfirmBackup: function() {
            var oModel = this._oBackupDialog.getModel("backup");
            var oData = oModel.getData();
            
            // Validate and sanitize input data
            const sanitizedData = this._sanitizeBackupData(oData);
            const validation = this._validateBackupData(sanitizedData);
            if (!validation.isValid) {
                MessageBox.error(validation.message);
                return;
            }
            
            this._oBackupDialog.setBusy(true);
            
            const ajaxConfig = this._securityUtils.createSecureAjaxConfig({
                url: "/a2a/agent8/v1/datasets/" + encodeURIComponent(sanitizedData.datasetName) + "/backup",
                type: "POST",
                data: JSON.stringify(sanitizedData),
                success: function(data) {
                    this._oBackupDialog.setBusy(false);
                    this._oBackupDialog.close();
                    
                    const safeBackupId = this._securityUtils.encodeHTML(data.backupId || 'N/A');
                    const safeBackupSize = this._securityUtils.encodeHTML(data.backupSize || '0');
                    const safeLocation = this._securityUtils.encodeHTML(data.backupLocation || 'N/A');
                    MessageBox.success(
                        "Backup created successfully!\\n" +
                        "Backup ID: " + safeBackupId + "\\n" +
                        "Backup size: " + safeBackupSize + "\\n" +
                        "Location: " + safeLocation
                    );
                    
                    this._extensionAPI.refresh();
                    this._securityUtils.auditLog('DATA_BACKUP_CREATED', { datasetName: sanitizedData.datasetName });
                }.bind(this),
                error: function(xhr) {
                    this._oBackupDialog.setBusy(false);
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    MessageBox.error("Backup failed: " + errorMsg);
                    this._securityUtils.auditLog('DATA_BACKUP_FAILED', { error: errorMsg });
                }.bind(this)
            });
            
            jQuery.ajax(ajaxConfig);
        },

        // Input validation and sanitization methods
        _validateStoreData: function(oData) {
            if (!oData.datasetName || !oData.datasetName.trim()) {
                return { isValid: false, message: "Dataset name is required" };
            }
            
            const datasetValidation = this._securityUtils.validateInput(oData.datasetName, 'datasetName', { required: true });
            if (!datasetValidation.isValid) {
                return { isValid: false, message: "Dataset name: " + datasetValidation.message };
            }
            
            return { isValid: true };
        },

        _sanitizeStoreData: function(oData) {
            return {
                datasetName: this._securityUtils.sanitizeInput(oData.datasetName),
                dataFormat: this._securityUtils.sanitizeInput(oData.dataFormat || 'JSON'),
                compressionType: this._securityUtils.sanitizeInput(oData.compressionType || 'GZIP'),
                encryptionEnabled: Boolean(oData.encryptionEnabled),
                createVersion: Boolean(oData.createVersion),
                cacheAfterStore: Boolean(oData.cacheAfterStore),
                validationEnabled: Boolean(oData.validationEnabled)
            };
        },

        _validateRetrieveData: function(oData) {
            if (!oData.datasetName || !oData.datasetName.trim()) {
                return { isValid: false, message: "Dataset name is required" };
            }
            
            const datasetValidation = this._securityUtils.validateInput(oData.datasetName, 'datasetName', { required: true });
            if (!datasetValidation.isValid) {
                return { isValid: false, message: "Dataset name: " + datasetValidation.message };
            }
            
            if (oData.maxRecords && oData.maxRecords < 0) {
                return { isValid: false, message: "Max records must be a positive number" };
            }
            
            return { isValid: true };
        },

        _sanitizeRetrieveData: function(oData) {
            return {
                datasetName: this._securityUtils.sanitizeInput(oData.datasetName),
                version: this._securityUtils.sanitizeInput(oData.version || ''),
                outputFormat: this._securityUtils.sanitizeInput(oData.outputFormat || 'JSON'),
                useCache: Boolean(oData.useCache),
                filterCriteria: this._sanitizeObject(oData.filterCriteria || {}),
                maxRecords: Math.max(0, parseInt(oData.maxRecords) || 0),
                includeMetadata: Boolean(oData.includeMetadata)
            };
        },

        _validateBackupData: function(oData) {
            if (!oData.datasetName || !oData.datasetName.trim()) {
                return { isValid: false, message: "Dataset name is required" };
            }
            
            const datasetValidation = this._securityUtils.validateInput(oData.datasetName, 'datasetName', { required: true });
            if (!datasetValidation.isValid) {
                return { isValid: false, message: "Dataset name: " + datasetValidation.message };
            }
            
            if (oData.retentionDays && (oData.retentionDays < 1 || oData.retentionDays > 3650)) {
                return { isValid: false, message: "Retention days must be between 1 and 3650" };
            }
            
            return { isValid: true };
        },

        _sanitizeBackupData: function(oData) {
            return {
                datasetName: this._securityUtils.sanitizeInput(oData.datasetName),
                backupType: this._securityUtils.sanitizeInput(oData.backupType || 'FULL'),
                compressionEnabled: Boolean(oData.compressionEnabled),
                encryptionEnabled: Boolean(oData.encryptionEnabled),
                backupLocation: this._securityUtils.sanitizeInput(oData.backupLocation || 'S3'),
                retentionDays: Math.max(1, Math.min(3650, parseInt(oData.retentionDays) || 90))
            };
        },

        _sanitizeObject: function(obj) {
            if (!obj || typeof obj !== 'object') return {};
            const sanitized = {};
            Object.keys(obj).forEach(key => {
                if (typeof obj[key] === 'string') {
                    sanitized[key] = this._securityUtils.sanitizeInput(obj[key]);
                } else if (typeof obj[key] === 'object' && obj[key] !== null) {
                    sanitized[key] = this._sanitizeObject(obj[key]);
                } else {
                    sanitized[key] = obj[key];
                }
            });
            return sanitized;
        },
        
        // Create Data Task Dialog Methods
        onCreateDataTask: function() {
            var oView = this.base.getView();
            
            if (!this._oCreateDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent8.ext.fragment.CreateDataTask",
                    controller: this
                }).then(function(oDialog) {
                    this._oCreateDialog = oDialog;
                    oView.addDependent(this._oCreateDialog);
                    this._oCreateDialog.setModel(this._oCreateModel, "create");
                    this._oCreateDialog.open();
                    
                    // Load available storage backends
                    this._loadAvailableBackends();
                    
                    this._securityUtils.auditLog('CREATE_DATA_DIALOG_OPENED', { action: 'create_data_task' });
                }.bind(this));
            } else {
                this._oCreateDialog.open();
                this._loadAvailableBackends();
            }
        },
        
        _loadAvailableBackends: function() {
            const ajaxConfig = this._securityUtils.createSecureAjaxConfig({
                url: "/a2a/agent8/v1/storage-backends",
                type: "GET",
                success: function(data) {
                    const sanitizedBackends = this._sanitizeArray(data || []);
                    this._oCreateModel.setProperty("/availableBackends", sanitizedBackends);
                }.bind(this),
                error: function(xhr) {
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    console.warn("Failed to load storage backends: " + errorMsg);
                    // Set empty array as fallback
                    this._oCreateModel.setProperty("/availableBackends", []);
                }.bind(this)
            });
            
            jQuery.ajax(ajaxConfig);
        },
        
        onCancelCreateTask: function() {
            this._oCreateDialog.close();
        },
        
        onConfirmCreateTask: function() {
            var oData = this._oCreateModel.getData();
            
            // Validate form
            if (!this._validateForm()) {
                MessageBox.error("Please correct the validation errors before creating the task.");
                return;
            }
            
            this._oCreateDialog.setBusy(true);
            
            var oSanitizedData = {
                taskName: this._securityUtils.sanitizeInput(oData.taskName),
                description: this._securityUtils.sanitizeInput(oData.description),
                datasetName: this._securityUtils.sanitizeInput(oData.datasetName),
                operationType: this._securityUtils.sanitizeInput(oData.operationType),
                priority: oData.priority,
                storageType: this._securityUtils.sanitizeInput(oData.storageType),
                storageBackend: this._securityUtils.sanitizeInput(oData.storageBackend),
                compressionEnabled: Boolean(oData.compressionEnabled),
                compressionType: this._securityUtils.sanitizeInput(oData.compressionType),
                encryptionEnabled: Boolean(oData.encryptionEnabled),
                encryptionType: this._securityUtils.sanitizeInput(oData.encryptionType),
                partitionStrategy: this._securityUtils.sanitizeInput(oData.partitionStrategy),
                cacheEnabled: Boolean(oData.cacheEnabled),
                autoWarmCache: Boolean(oData.autoWarmCache),
                cacheOnWrite: Boolean(oData.cacheOnWrite),
                cacheLevel: this._securityUtils.sanitizeInput(oData.cacheLevel),
                cacheTTL: Math.max(1, Math.min(1440, parseInt(oData.cacheTTL) || 60)),
                evictionPolicy: this._securityUtils.sanitizeInput(oData.evictionPolicy),
                memoryCacheSize: Math.max(0.1, Math.min(64, parseFloat(oData.memoryCacheSize) || 2)),
                redisCacheSize: Math.max(0.1, Math.min(256, parseFloat(oData.redisCacheSize) || 8)),
                versioningEnabled: Boolean(oData.versioningEnabled),
                autoVersioning: Boolean(oData.autoVersioning),
                incrementalBackup: Boolean(oData.incrementalBackup),
                versionStrategy: this._securityUtils.sanitizeInput(oData.versionStrategy),
                retentionDays: Math.max(1, Math.min(365, parseInt(oData.retentionDays) || 90)),
                maxVersions: Math.max(1, Math.min(100, parseInt(oData.maxVersions) || 10)),
                checksumValidation: Boolean(oData.checksumValidation),
                checksumAlgorithm: this._securityUtils.sanitizeInput(oData.checksumAlgorithm),
                batchSize: Math.max(100, Math.min(100000, parseInt(oData.batchSize) || 1000)),
                parallelProcessing: Boolean(oData.parallelProcessing),
                threadPoolSize: Math.max(1, Math.min(32, parseInt(oData.threadPoolSize) || 4)),
                connectionPoolSize: Math.max(1, Math.min(100, parseInt(oData.connectionPoolSize) || 10)),
                connectionTimeout: Math.max(5, Math.min(300, parseInt(oData.connectionTimeout) || 30)),
                autoOptimization: Boolean(oData.autoOptimization),
                indexOptimization: Boolean(oData.indexOptimization),
                queryOptimization: Boolean(oData.queryOptimization),
                memoryOptimization: Boolean(oData.memoryOptimization),
                performanceMonitoring: Boolean(oData.performanceMonitoring),
                trackThroughput: Boolean(oData.trackThroughput),
                trackResponseTimes: Boolean(oData.trackResponseTimes),
                trackResourceUsage: Boolean(oData.trackResourceUsage)
            };
            
            const ajaxConfig = this._securityUtils.createSecureAjaxConfig({
                url: "/a2a/agent8/v1/tasks",
                type: "POST",
                data: JSON.stringify(oSanitizedData),
                success: function(data) {
                    this._oCreateDialog.setBusy(false);
                    this._oCreateDialog.close();
                    MessageToast.show("Data management task created successfully");
                    this._extensionAPI.refresh();
                    
                    this._securityUtils.auditLog('DATA_TASK_CREATED', { taskName: oSanitizedData.taskName });
                }.bind(this),
                error: function(xhr) {
                    this._oCreateDialog.setBusy(false);
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    MessageBox.error("Failed to create data task: " + errorMsg);
                    this._securityUtils.auditLog('DATA_TASK_CREATE_FAILED', { error: errorMsg });
                }.bind(this)
            });
            
            jQuery.ajax(ajaxConfig);
        },
        
        // Validation Event Handlers
        onTaskNameChange: function() {
            var sValue = this._oCreateModel.getProperty("/taskName");
            var oValidation = this._validateTaskName(sValue);
            
            this._oCreateModel.setProperty("/taskNameState", oValidation.state);
            this._oCreateModel.setProperty("/taskNameStateText", oValidation.message);
        },
        
        onDatasetNameChange: function() {
            var sValue = this._oCreateModel.getProperty("/datasetName");
            var oValidation = this._validateDatasetName(sValue);
            
            this._oCreateModel.setProperty("/datasetNameState", oValidation.state);
            this._oCreateModel.setProperty("/datasetNameStateText", oValidation.message);
        },
        
        onOperationTypeChange: function() {
            var sValue = this._oCreateModel.getProperty("/operationType");
            var oValidation = this._validateOperationType(sValue);
            
            this._oCreateModel.setProperty("/operationTypeState", oValidation.state);
            this._oCreateModel.setProperty("/operationTypeStateText", oValidation.message);
            
            // Auto-suggest configurations based on operation type
            this._applyOperationDefaults(sValue);
        },
        
        _applyOperationDefaults: function(sOperationType) {
            switch (sOperationType) {
                case "STORE":
                    this._oCreateModel.setProperty("/compressionEnabled", true);
                    this._oCreateModel.setProperty("/encryptionEnabled", true);
                    this._oCreateModel.setProperty("/versioningEnabled", true);
                    this._oCreateModel.setProperty("/cacheEnabled", false);
                    break;
                case "RETRIEVE":
                    this._oCreateModel.setProperty("/cacheEnabled", true);
                    this._oCreateModel.setProperty("/autoWarmCache", true);
                    this._oCreateModel.setProperty("/performanceMonitoring", true);
                    break;
                case "BACKUP":
                    this._oCreateModel.setProperty("/compressionEnabled", true);
                    this._oCreateModel.setProperty("/encryptionEnabled", true);
                    this._oCreateModel.setProperty("/checksumValidation", true);
                    break;
                case "CACHE":
                    this._oCreateModel.setProperty("/cacheEnabled", true);
                    this._oCreateModel.setProperty("/autoWarmCache", false);
                    this._oCreateModel.setProperty("/cacheOnWrite", true);
                    break;
                case "COMPRESS":
                    this._oCreateModel.setProperty("/compressionEnabled", true);
                    this._oCreateModel.setProperty("/compressionType", "ZSTD");
                    break;
                default:
                    // Keep current values
                    break;
            }
        },
        
        onSelectDataset: function() {
            MessageBox.information("Dataset selection functionality will be available in a future update.");
        },
        
        // Validation Methods
        _validateTaskName: function(sValue) {
            if (!sValue || sValue.trim().length === 0) {
                return { state: "Error", message: "Task name is required" };
            }
            if (sValue.length < 3) {
                return { state: "Warning", message: "Task name should be at least 3 characters" };
            }
            if (sValue.length > 100) {
                return { state: "Error", message: "Task name must not exceed 100 characters" };
            }
            if (!/^[a-zA-Z0-9\s\-_\.]+$/.test(sValue)) {
                return { state: "Error", message: "Task name contains invalid characters" };
            }
            return { state: "Success", message: "" };
        },
        
        _validateDatasetName: function(sValue) {
            if (!sValue || sValue.trim().length === 0) {
                return { state: "Error", message: "Dataset name is required" };
            }
            if (sValue.length < 2) {
                return { state: "Warning", message: "Dataset name should be at least 2 characters" };
            }
            if (sValue.length > 50) {
                return { state: "Error", message: "Dataset name must not exceed 50 characters" };
            }
            if (!/^[a-zA-Z0-9\-_]+$/.test(sValue)) {
                return { state: "Error", message: "Dataset name can only contain letters, numbers, hyphens, and underscores" };
            }
            return { state: "Success", message: "" };
        },
        
        _validateOperationType: function(sValue) {
            if (!sValue || sValue.trim().length === 0) {
                return { state: "Error", message: "Operation type is required" };
            }
            const allowedOperations = ['STORE', 'RETRIEVE', 'UPDATE', 'DELETE', 'BACKUP', 'RESTORE', 'MIGRATE', 'COMPRESS', 'CACHE', 'VALIDATE'];
            if (!allowedOperations.includes(sValue)) {
                return { state: "Error", message: "Invalid operation type" };
            }
            return { state: "Success", message: "" };
        },
        
        _validateForm: function() {
            var oData = this._oCreateModel.getData();
            var bValid = true;
            
            // Validate task name
            var oTaskNameValidation = this._validateTaskName(oData.taskName);
            this._oCreateModel.setProperty("/taskNameState", oTaskNameValidation.state);
            this._oCreateModel.setProperty("/taskNameStateText", oTaskNameValidation.message);
            if (oTaskNameValidation.state === "Error") bValid = false;
            
            // Validate dataset name
            var oDatasetNameValidation = this._validateDatasetName(oData.datasetName);
            this._oCreateModel.setProperty("/datasetNameState", oDatasetNameValidation.state);
            this._oCreateModel.setProperty("/datasetNameStateText", oDatasetNameValidation.message);
            if (oDatasetNameValidation.state === "Error") bValid = false;
            
            // Validate operation type
            var oOperationTypeValidation = this._validateOperationType(oData.operationType);
            this._oCreateModel.setProperty("/operationTypeState", oOperationTypeValidation.state);
            this._oCreateModel.setProperty("/operationTypeStateText", oOperationTypeValidation.message);
            if (oOperationTypeValidation.state === "Error") bValid = false;
            
            return bValid;
        },
        
        _sanitizeArray: function(arr) {
            if (!Array.isArray(arr)) return [];
            return arr.map(item => {
                if (typeof item === 'string') {
                    return this._securityUtils.sanitizeInput(item);
                } else if (typeof item === 'object') {
                    return this._sanitizeObject(item);
                } else {
                    return item;
                }
            });
        }
    });
});