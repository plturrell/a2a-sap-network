sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/ui/core/Fragment",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/model/json/JSONModel"
], function (ControllerExtension, Fragment, MessageBox, MessageToast, JSONModel) {
    "use strict";

    return ControllerExtension.extend("a2a.network.agent8.ext.controller.ListReportExt", {
        
        override: {
            onInit: function () {
                this._extensionAPI = this.base.getExtensionAPI();
                this._startRealtimeUpdates();
            }
        },

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
                    
                    var oModel = new JSONModel({
                        taskName: "",
                        description: "",
                        datasetName: "",
                        operationType: "",
                        storageType: "HANA",
                        storageBackend: "",
                        priority: "MEDIUM",
                        compressionEnabled: true,
                        encryptionEnabled: false,
                        cacheEnabled: true,
                        versioningEnabled: true
                    });
                    this._oCreateDialog.setModel(oModel, "create");
                    this._oCreateDialog.open();
                    this._loadStorageOptions();
                }.bind(this));
            } else {
                this._oCreateDialog.open();
                this._loadStorageOptions();
            }
        },

        _loadStorageOptions: function() {
            jQuery.ajax({
                url: "/a2a/agent8/v1/storage-options",
                type: "GET",
                success: function(data) {
                    var oModel = this._oCreateDialog.getModel("create");
                    var oData = oModel.getData();
                    oData.availableBackends = data.backends;
                    oData.storageTypes = data.types;
                    oData.availableDatasets = data.datasets;
                    oModel.setData(oData);
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to load storage options: " + xhr.responseText);
                }.bind(this)
            });
        },

        onDataDashboard: function() {
            var oView = this.base.getView();
            
            if (!this._oDashboard) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent8.ext.fragment.DataDashboard",
                    controller: this
                }).then(function(oDialog) {
                    this._oDashboard = oDialog;
                    oView.addDependent(this._oDashboard);
                    this._oDashboard.open();
                    this._loadDashboardData();
                }.bind(this));
            } else {
                this._oDashboard.open();
                this._loadDashboardData();
            }
        },

        _loadDashboardData: function() {
            this._oDashboard.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent8/v1/dashboard",
                type: "GET",
                success: function(data) {
                    this._oDashboard.setBusy(false);
                    
                    var oDashboardModel = new JSONModel({
                        summary: data.summary,
                        storageMetrics: data.storageMetrics,
                        cacheMetrics: data.cacheMetrics,
                        performanceMetrics: data.performanceMetrics,
                        trends: data.trends,
                        alerts: data.alerts
                    });
                    
                    this._oDashboard.setModel(oDashboardModel, "dashboard");
                    this._createDashboardCharts(data);
                }.bind(this),
                error: function(xhr) {
                    this._oDashboard.setBusy(false);
                    MessageBox.error("Failed to load dashboard data: " + xhr.responseText);
                }.bind(this)
            });
        },

        _createDashboardCharts: function(data) {
            this._createStorageUtilizationChart(data.storageMetrics);
            this._createCachePerformanceChart(data.cacheMetrics);
            this._createThroughputTrendsChart(data.trends);
        },

        _createStorageUtilizationChart: function(storageData) {
            var oVizFrame = this._oDashboard.byId("storageUtilizationChart");
            if (!oVizFrame || !storageData) return;
            
            var aChartData = storageData.backends.map(function(backend) {
                return {
                    Backend: backend.name,
                    Used: backend.usedSpace,
                    Available: backend.capacity - backend.usedSpace,
                    Utilization: backend.utilization
                };
            });
            
            var oChartModel = new sap.ui.model.json.JSONModel({
                storageData: aChartData
            });
            oVizFrame.setModel(oChartModel);
            
            oVizFrame.setVizProperties({
                categoryAxis: {
                    title: { text: "Storage Backends" }
                },
                valueAxis: {
                    title: { text: "Storage (GB)" }
                },
                title: {
                    text: "Storage Utilization by Backend"
                }
            });
        },

        onStorageManager: function() {
            var oView = this.base.getView();
            
            if (!this._oStorageDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent8.ext.fragment.StorageManager",
                    controller: this
                }).then(function(oDialog) {
                    this._oStorageDialog = oDialog;
                    oView.addDependent(this._oStorageDialog);
                    this._oStorageDialog.open();
                    this._loadStorageData();
                }.bind(this));
            } else {
                this._oStorageDialog.open();
                this._loadStorageData();
            }
        },

        _loadStorageData: function() {
            jQuery.ajax({
                url: "/a2a/agent8/v1/storage-backends",
                type: "GET",
                success: function(data) {
                    var oModel = new JSONModel({
                        backends: data.backends,
                        totalCapacity: data.totalCapacity,
                        totalUsed: data.totalUsed,
                        utilizationRate: data.utilizationRate
                    });
                    this._oStorageDialog.setModel(oModel, "storage");
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to load storage data: " + xhr.responseText);
                }.bind(this)
            });
        },

        onCacheManager: function() {
            var oView = this.base.getView();
            
            if (!this._oCacheDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent8.ext.fragment.CacheManager",
                    controller: this
                }).then(function(oDialog) {
                    this._oCacheDialog = oDialog;
                    oView.addDependent(this._oCacheDialog);
                    this._oCacheDialog.open();
                    this._loadCacheData();
                }.bind(this));
            } else {
                this._oCacheDialog.open();
                this._loadCacheData();
            }
        },

        _loadCacheData: function() {
            jQuery.ajax({
                url: "/a2a/agent8/v1/cache-status",
                type: "GET",
                success: function(data) {
                    var oModel = new JSONModel({
                        memoryCache: data.memoryCache,
                        redisCache: data.redisCache,
                        hitRates: data.hitRates,
                        evictionStats: data.evictionStats,
                        topEntries: data.topEntries
                    });
                    this._oCacheDialog.setModel(oModel, "cache");
                    this._createCacheVisualizations(data);
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to load cache data: " + xhr.responseText);
                }.bind(this)
            });
        },

        _createCacheVisualizations: function(data) {
            var oCacheChart = this._oCacheDialog.byId("cacheHitRateChart");
            if (!oCacheChart || !data.hitRates) return;
            
            var aChartData = data.hitRates.map(function(rate) {
                return {
                    Time: rate.timestamp,
                    MemoryHitRate: rate.memoryHitRate,
                    RedisHitRate: rate.redisHitRate
                };
            });
            
            var oChartModel = new sap.ui.model.json.JSONModel({
                hitRateData: aChartData
            });
            oCacheChart.setModel(oChartModel);
        },

        onVersionManager: function() {
            var oView = this.base.getView();
            
            if (!this._oVersionDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent8.ext.fragment.VersionManager",
                    controller: this
                }).then(function(oDialog) {
                    this._oVersionDialog = oDialog;
                    oView.addDependent(this._oVersionDialog);
                    this._oVersionDialog.open();
                    this._loadVersionData();
                }.bind(this));
            } else {
                this._oVersionDialog.open();
                this._loadVersionData();
            }
        },

        _loadVersionData: function() {
            jQuery.ajax({
                url: "/a2a/agent8/v1/version-history",
                type: "GET",
                success: function(data) {
                    var oModel = new JSONModel({
                        versions: data.versions,
                        versionStats: data.stats,
                        retentionPolicy: data.retentionPolicy,
                        storageUsage: data.storageUsage
                    });
                    this._oVersionDialog.setModel(oModel, "version");
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to load version data: " + xhr.responseText);
                }.bind(this)
            });
        },

        onBulkOperations: function() {
            var oTable = this._extensionAPI.getTable();
            var aSelectedContexts = oTable.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageBox.warning("Please select at least one dataset for bulk operations.");
                return;
            }
            
            var oView = this.base.getView();
            
            if (!this._oBulkDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent8.ext.fragment.BulkOperations",
                    controller: this
                }).then(function(oDialog) {
                    this._oBulkDialog = oDialog;
                    oView.addDependent(this._oBulkDialog);
                    
                    var aDatasetNames = aSelectedContexts.map(function(oContext) {
                        return oContext.getProperty("datasetName");
                    });
                    
                    var oModel = new JSONModel({
                        selectedDatasets: aDatasetNames,
                        operation: "",
                        parameters: {},
                        parallelExecution: true,
                        batchSize: 1000,
                        continueOnError: false
                    });
                    this._oBulkDialog.setModel(oModel, "bulk");
                    this._oBulkDialog.open();
                }.bind(this));
            } else {
                this._oBulkDialog.open();
            }
        },

        onDataImport: function() {
            var oView = this.base.getView();
            
            if (!this._oImportDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent8.ext.fragment.DataImport",
                    controller: this
                }).then(function(oDialog) {
                    this._oImportDialog = oDialog;
                    oView.addDependent(this._oImportDialog);
                    
                    var oModel = new JSONModel({
                        importType: "FILE",
                        filePath: "",
                        dataFormat: "JSON",
                        targetDataset: "",
                        createNewDataset: true,
                        validateData: true,
                        compressionEnabled: true,
                        batchSize: 5000
                    });
                    this._oImportDialog.setModel(oModel, "import");
                    this._oImportDialog.open();
                }.bind(this));
            } else {
                this._oImportDialog.open();
            }
        },

        onDataExport: function() {
            var oView = this.base.getView();
            
            if (!this._oExportDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent8.ext.fragment.DataExport",
                    controller: this
                }).then(function(oDialog) {
                    this._oExportDialog = oDialog;
                    oView.addDependent(this._oExportDialog);
                    
                    var oModel = new JSONModel({
                        exportFormat: "JSON",
                        includeMetadata: true,
                        compressionEnabled: true,
                        outputPath: "",
                        filterCriteria: {},
                        maxRecords: 0
                    });
                    this._oExportDialog.setModel(oModel, "export");
                    this._oExportDialog.open();
                    this._loadExportOptions();
                }.bind(this));
            } else {
                this._oExportDialog.open();
                this._loadExportOptions();
            }
        },

        _loadExportOptions: function() {
            jQuery.ajax({
                url: "/a2a/agent8/v1/export-options",
                type: "GET",
                success: function(data) {
                    var oModel = this._oExportDialog.getModel("export");
                    var oData = oModel.getData();
                    oData.availableDatasets = data.datasets;
                    oData.exportFormats = data.formats;
                    oModel.setData(oData);
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to load export options: " + xhr.responseText);
                }.bind(this)
            });
        },

        onBackupManager: function() {
            var oView = this.base.getView();
            
            if (!this._oBackupDialog) {
                Fragment.load({
                    id: oView.getId(),
                    name: "a2a.network.agent8.ext.fragment.BackupManager",
                    controller: this
                }).then(function(oDialog) {
                    this._oBackupDialog = oDialog;
                    oView.addDependent(this._oBackupDialog);
                    this._oBackupDialog.open();
                    this._loadBackupData();
                }.bind(this));
            } else {
                this._oBackupDialog.open();
                this._loadBackupData();
            }
        },

        _loadBackupData: function() {
            jQuery.ajax({
                url: "/a2a/agent8/v1/backup-status",
                type: "GET",
                success: function(data) {
                    var oModel = new JSONModel({
                        backups: data.backups,
                        schedule: data.schedule,
                        storageUsage: data.storageUsage,
                        retentionPolicy: data.retentionPolicy
                    });
                    this._oBackupDialog.setModel(oModel, "backup");
                }.bind(this),
                error: function(xhr) {
                    MessageBox.error("Failed to load backup data: " + xhr.responseText);
                }.bind(this)
            });
        },

        _startRealtimeUpdates: function() {
            this._realtimeEventSource = new EventSource("/a2a/agent8/v1/realtime-updates");
            
            this._realtimeEventSource.onmessage = function(event) {
                var data = JSON.parse(event.data);
                
                if (data.type === "storage_alert") {
                    MessageToast.show("Storage Alert: " + data.message);
                } else if (data.type === "cache_stats") {
                    this._updateCacheStats(data.stats);
                } else if (data.type === "operation_complete") {
                    MessageToast.show("Operation completed: " + data.operation);
                    this._extensionAPI.refresh();
                }
            }.bind(this);
            
            this._realtimeEventSource.onerror = function() {
                MessageToast.show("Real-time updates disconnected");
            }.bind(this);
        },

        _updateCacheStats: function(stats) {
            // Update cache statistics in real-time if cache dialog is open
            if (this._oCacheDialog && this._oCacheDialog.isOpen()) {
                var oModel = this._oCacheDialog.getModel("cache");
                if (oModel) {
                    var oData = oModel.getData();
                    oData.hitRates = stats.hitRates;
                    oModel.setData(oData);
                }
            }
        },

        onConfirmCreateTask: function() {
            var oModel = this._oCreateDialog.getModel("create");
            var oData = oModel.getData();
            
            if (!oData.taskName || !oData.datasetName || !oData.operationType) {
                MessageBox.error("Please fill all required fields");
                return;
            }
            
            this._oCreateDialog.setBusy(true);
            
            jQuery.ajax({
                url: "/a2a/agent8/v1/data-tasks",
                type: "POST",
                contentType: "application/json",
                data: JSON.stringify(oData),
                success: function(data) {
                    this._oCreateDialog.setBusy(false);
                    this._oCreateDialog.close();
                    MessageToast.show("Data management task created successfully");
                    this._extensionAPI.refresh();
                }.bind(this),
                error: function(xhr) {
                    this._oCreateDialog.setBusy(false);
                    MessageBox.error("Failed to create task: " + xhr.responseText);
                }.bind(this)
            });
        },

        onCancelCreateTask: function() {
            this._oCreateDialog.close();
        }
    });
});