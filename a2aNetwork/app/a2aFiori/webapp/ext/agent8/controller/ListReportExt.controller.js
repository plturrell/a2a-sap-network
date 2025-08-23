sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/ui/core/Fragment",
    "sap/m/MessageBox",
    "sap/m/MessageToast",
    "sap/ui/model/json/JSONModel",
    "a2a/network/agent8/ext/utils/SecurityUtils"
], function (ControllerExtension, Fragment, MessageBox, MessageToast, JSONModel, SecurityUtils) {
    "use strict";

    /**
     * @class a2a.network.agent8.ext.controller.ListReportExt
     * @extends sap.ui.core.mvc.ControllerExtension
     * @description Controller extension for Agent 8 List Report - Data Management and Storage Operations.
     * Provides comprehensive data storage, caching, versioning, backup, and import/export capabilities
     * with enterprise-grade security, performance optimization, and accessibility features.
     */
    return ControllerExtension.extend("a2a.network.agent8.ext.controller.ListReportExt", {
        
        override: {
            /**
             * @function onInit
             * @description Initializes the controller extension with security utilities, device model, and real-time updates.
             * @override
             */
            onInit: function () {
                this._extensionAPI = this.base.getExtensionAPI();
                this._securityUtils = SecurityUtils;
                this._initializeDeviceModel();
                this._initializeDialogCache();
                this._initializePerformanceOptimizations();
                this._startRealtimeUpdates();
            },
            
            /**
             * @function onExit
             * @description Cleanup resources on controller destruction.
             * @override
             */
            onExit: function() {
                this._cleanupResources();
                if (this.base.onExit) {
                    this.base.onExit.apply(this, arguments);
                }
            }
        },
        
        // Dialog caching for performance
        _dialogCache: {},
        
        // Error recovery configuration
        _errorRecoveryConfig: {
            maxRetries: 3,
            retryDelay: 1000,
            exponentialBackoff: true
        },

        /**
         * @function _initializeDeviceModel
         * @description Sets up device model for responsive design.
         * @private
         */
        _initializeDeviceModel: function() {
            var oDeviceModel = new sap.ui.model.json.JSONModel(sap.ui.Device);
            this.base.getView().setModel(oDeviceModel, "device");
        },
        
        /**
         * @function _initializeDialogCache
         * @description Initializes dialog cache for performance.
         * @private
         */
        _initializeDialogCache: function() {
            this._dialogCache = {};
        },
        
        /**
         * @function _initializePerformanceOptimizations
         * @description Sets up performance optimization features.
         * @private
         */
        _initializePerformanceOptimizations: function() {
            // Throttle dashboard updates
            this._throttledDashboardUpdate = this._throttle(this._loadDashboardData.bind(this), 1000);
            // Debounce search operations
            this._debouncedSearch = this._debounce(this._performSearch.bind(this), 300);
        },
        
        /**
         * @function _throttle
         * @description Creates a throttled function.
         * @param {Function} fn - Function to throttle
         * @param {number} limit - Time limit in milliseconds
         * @returns {Function} Throttled function
         * @private
         */
        _throttle: function(fn, limit) {
            var inThrottle;
            return function() {
                var args = arguments;
                var context = this;
                if (!inThrottle) {
                    fn.apply(context, args);
                    inThrottle = true;
                    setTimeout(function() { inThrottle = false; }, limit);
                }
            };
        },
        
        /**
         * @function _debounce
         * @description Creates a debounced function.
         * @param {Function} fn - Function to debounce
         * @param {number} delay - Delay in milliseconds
         * @returns {Function} Debounced function
         * @private
         */
        _debounce: function(fn, delay) {
            var timeoutId;
            return function() {
                var context = this;
                var args = arguments;
                clearTimeout(timeoutId);
                timeoutId = setTimeout(function() {
                    fn.apply(context, args);
                }, delay);
            };
        },
        
        /**
         * @function _performSearch
         * @description Performs search operation (placeholder for search functionality).
         * @param {string} sQuery - Search query
         * @private
         */
        _performSearch: function(sQuery) {
            // Implement search logic
        },
        
        /**
         * @function onCreateDataTask
         * @description Opens dialog to create new data management task.
         * @public
         */
        onCreateDataTask: function() {
            var oView = this.base.getView();
            
            this._getOrCreateDialog("createDataTask", "a2a.network.agent8.ext.fragment.CreateDataTask")
                .then(function(oDialog) {
                    
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
                    oDialog.setModel(oModel, "create");
                    oDialog.open();
                    this._loadStorageOptions(oDialog);
                }.bind(this));
        },
        
        /**
         * @function _getOrCreateDialog
         * @description Gets cached dialog or creates new one.
         * @param {string} sDialogId - Dialog identifier
         * @param {string} sFragmentName - Fragment name
         * @returns {Promise<sap.m.Dialog>} Promise resolving to dialog
         * @private
         */
        _getOrCreateDialog: function(sDialogId, sFragmentName) {
            var that = this;
            
            if (this._dialogCache[sDialogId]) {
                return Promise.resolve(this._dialogCache[sDialogId]);
            }
            
            return Fragment.load({
                id: this.base.getView().getId(),
                name: sFragmentName,
                controller: this
            }).then(function(oDialog) {
                that._dialogCache[sDialogId] = oDialog;
                that.base.getView().addDependent(oDialog);
                
                // Enable accessibility
                that._enableDialogAccessibility(oDialog);
                
                // Optimize for mobile
                that._optimizeDialogForDevice(oDialog);
                
                return oDialog;
            });
        },
        
        /**
         * @function _enableDialogAccessibility
         * @description Adds accessibility features to dialog.
         * @param {sap.m.Dialog} oDialog - Dialog to enhance
         * @private
         */
        _enableDialogAccessibility: function(oDialog) {
            oDialog.addEventDelegate({
                onAfterRendering: function() {
                    var $dialog = oDialog.$();
                    
                    // Set tabindex for focusable elements
                    $dialog.find("input, button, select, textarea").attr("tabindex", "0");
                    
                    // Handle escape key
                    $dialog.on("keydown", function(e) {
                        if (e.key === "Escape") {
                            oDialog.close();
                        }
                    });
                    
                    // Focus first input on open
                    setTimeout(function() {
                        $dialog.find("input:visible:first").focus();
                    }, 100);
                }
            });
        },
        
        /**
         * @function _optimizeDialogForDevice
         * @description Optimizes dialog for current device.
         * @param {sap.m.Dialog} oDialog - Dialog to optimize
         * @private
         */
        _optimizeDialogForDevice: function(oDialog) {
            if (sap.ui.Device.system.phone) {
                oDialog.setStretch(true);
                oDialog.setContentWidth("100%");
                oDialog.setContentHeight("100%");
            } else if (sap.ui.Device.system.tablet) {
                oDialog.setContentWidth("95%");
                oDialog.setContentHeight("90%");
            }
        },
        
        /**
         * @function _withErrorRecovery
         * @description Wraps operation with error recovery.
         * @param {Function} fnOperation - Operation to execute
         * @param {Object} oOptions - Recovery options
         * @returns {Promise} Promise with error recovery
         * @private
         */
        _withErrorRecovery: function(fnOperation, oOptions) {
            var that = this;
            var oConfig = Object.assign({}, this._errorRecoveryConfig, oOptions);
            
            function attempt(retriesLeft, delay) {
                return fnOperation().catch(function(error) {
                    if (retriesLeft > 0) {
                        var oBundle = that.base.getView().getModel("i18n").getResourceBundle();
                        var sRetryMsg = oBundle.getText("recovery.retrying") || "Network error. Retrying...";
                        MessageToast.show(sRetryMsg);
                        
                        return new Promise(function(resolve) {
                            setTimeout(resolve, delay);
                        }).then(function() {
                            var nextDelay = oConfig.exponentialBackoff ? delay * 2 : delay;
                            return attempt(retriesLeft - 1, nextDelay);
                        });
                    }
                    throw error;
                });
            }
            
            return attempt(oConfig.maxRetries, oConfig.retryDelay);
        },
        
        /**
         * @function _cleanupResources
         * @description Cleans up resources to prevent memory leaks.
         * @private
         */
        _cleanupResources: function() {
            // Clean up event sources
            if (this._realtimeEventSource) {
                this._realtimeEventSource.close();
                this._realtimeEventSource = null;
            }
            if (this._optimizationEventSource) {
                this._optimizationEventSource.close();
                this._optimizationEventSource = null;
            }
            if (this._batchMonitoringEventSource) {
                this._batchMonitoringEventSource.close();
                this._batchMonitoringEventSource = null;
            }
            
            // Clean up cached dialogs
            Object.keys(this._dialogCache).forEach(function(key) {
                if (this._dialogCache[key]) {
                    this._dialogCache[key].destroy();
                }
            }.bind(this));
            this._dialogCache = {};
        },

        /**
         * @function _loadStorageOptions
         * @description Loads storage backend options with error recovery.
         * @param {sap.m.Dialog} oDialog - Target dialog (optional)
         * @private
         */
        _loadStorageOptions: function(oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["createDataTask"];
            if (!oTargetDialog) return;
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    jQuery.ajax({
                        url: "/a2a/agent8/v1/storage-options",
                        type: "GET",
                        success: function(data) {
                            var oModel = oTargetDialog.getModel("create");
                            var oData = oModel.getData();
                            oData.availableBackends = data.backends;
                            oData.storageTypes = data.types;
                            oData.availableDatasets = data.datasets;
                            oModel.setData(oData);
                            resolve(data);
                        },
                        error: function(xhr) {
                            const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                            reject(new Error("Failed to load storage options: " + errorMsg));
                        }
                    });
                }.bind(this));
            }.bind(this)).catch(function(error) {
                MessageBox.error(error.message);
            });
        },

        /**
         * @function onDataDashboard
         * @description Opens comprehensive data management dashboard.
         * @public
         */
        onDataDashboard: function() {
            this._getOrCreateDialog("dataDashboard", "a2a.network.agent8.ext.fragment.DataDashboard")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadDashboardData(oDialog);
                }.bind(this));
        },

        /**
         * @function _loadDashboardData
         * @description Loads dashboard data with metrics and charts.
         * @param {sap.m.Dialog} oDialog - Target dialog (optional)
         * @private
         */
        _loadDashboardData: function(oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["dataDashboard"];
            if (!oTargetDialog) return;
            
            oTargetDialog.setBusy(true);
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    jQuery.ajax({
                        url: "/a2a/agent8/v1/dashboard",
                        type: "GET",
                        success: function(data) {
                            var oDashboardModel = new JSONModel({
                                summary: data.summary,
                                storageMetrics: data.storageMetrics,
                                cacheMetrics: data.cacheMetrics,
                                performanceMetrics: data.performanceMetrics,
                                trends: data.trends,
                                alerts: data.alerts
                            });
                            
                            oTargetDialog.setModel(oDashboardModel, "dashboard");
                            resolve(data);
                        },
                        error: function(xhr) {
                            const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                            reject(new Error("Failed to load dashboard data: " + errorMsg));
                        }
                    });
                });
            }).then(function(data) {
                oTargetDialog.setBusy(false);
                this._createDashboardCharts(data, oTargetDialog);
            }.bind(this)).catch(function(error) {
                oTargetDialog.setBusy(false);
                MessageBox.error(error.message);
            });
        },

        /**
         * @function _createDashboardCharts
         * @description Creates dashboard visualization charts.
         * @param {Object} data - Dashboard data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createDashboardCharts: function(data, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["dataDashboard"];
            if (!oTargetDialog) return;
            
            this._createStorageUtilizationChart(data.storageMetrics, oTargetDialog);
            this._createCachePerformanceChart(data.cacheMetrics, oTargetDialog);
            this._createThroughputTrendsChart(data.trends, oTargetDialog);
        },

        /**
         * @function _createStorageUtilizationChart
         * @description Creates storage utilization visualization chart.
         * @param {Object} storageData - Storage metrics data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createStorageUtilizationChart: function(storageData, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["dataDashboard"];
            if (!oTargetDialog) return;
            
            var oVizFrame = oTargetDialog.byId("storageUtilizationChart");
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
            
            var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
            oVizFrame.setVizProperties({
                categoryAxis: {
                    title: { text: oBundle.getText("chart.storageBackends") || "Storage Backends" }
                },
                valueAxis: {
                    title: { text: oBundle.getText("chart.storageGB") || "Storage (GB)" }
                },
                title: {
                    text: oBundle.getText("chart.storageUtilization") || "Storage Utilization by Backend"
                }
            });
        },

        /**
         * @function onStorageManager
         * @description Opens storage backend management interface.
         * @public
         */
        onStorageManager: function() {
            this._getOrCreateDialog("storageManager", "a2a.network.agent8.ext.fragment.StorageManager")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadStorageData(oDialog);
                }.bind(this));
        },

        /**
         * @function _loadStorageData
         * @description Loads storage backend information.
         * @param {sap.m.Dialog} oDialog - Target dialog (optional)
         * @private
         */
        _loadStorageData: function(oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["storageManager"];
            if (!oTargetDialog) return;
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
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
                            oTargetDialog.setModel(oModel, "storage");
                            resolve(data);
                        },
                        error: function(xhr) {
                            const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                            reject(new Error("Failed to load storage data: " + errorMsg));
                        }
                    });
                });
            }).catch(function(error) {
                MessageBox.error(error.message);
            });
        },

        /**
         * @function onCacheManager
         * @description Opens cache management interface.
         * @public
         */
        onCacheManager: function() {
            this._getOrCreateDialog("cacheManager", "a2a.network.agent8.ext.fragment.CacheManager")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadCacheData(oDialog);
                }.bind(this));
        },

        /**
         * @function _loadCacheData
         * @description Loads cache performance and statistics data.
         * @param {sap.m.Dialog} oDialog - Target dialog (optional)
         * @private
         */
        _loadCacheData: function(oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["cacheManager"];
            if (!oTargetDialog) return;
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
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
                            oTargetDialog.setModel(oModel, "cache");
                            resolve(data);
                        },
                        error: function(xhr) {
                            const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                            reject(new Error("Failed to load cache data: " + errorMsg));
                        }
                    });
                });
            }).then(function(data) {
                this._createCacheVisualizations(data, oTargetDialog);
            }.bind(this)).catch(function(error) {
                MessageBox.error(error.message);
            });
        },

        /**
         * @function _createCacheVisualizations
         * @description Creates cache performance visualization charts.
         * @param {Object} data - Cache metrics data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createCacheVisualizations: function(data, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["cacheManager"];
            if (!oTargetDialog) return;
            
            var oCacheChart = oTargetDialog.byId("cacheHitRateChart");
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
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    MessageBox.error("Failed to load version data: " + errorMsg);
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
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    MessageBox.error("Failed to load export options: " + errorMsg);
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
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    MessageBox.error("Failed to load backup data: " + errorMsg);
                }.bind(this)
            });
        }
        
        /**
         * @function _createCachePerformanceChart
         * @description Creates cache performance visualization chart.
         * @param {Object} data - Cache performance data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createCachePerformanceChart: function(data, oDialog) {
            // Implementation placeholder for cache performance chart creation
            // This would create charts for cache hit rates, eviction patterns, etc.
        },
        
        /**
         * @function _createThroughputTrendsChart
         * @description Creates throughput trends visualization chart.
         * @param {Object} data - Throughput data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createThroughputTrendsChart: function(data, oDialog) {
            // Implementation placeholder for throughput trends chart creation
            // This would create charts for read/write throughput over time
        },

        /**
         * @function _updateCacheStats
         * @description Updates cache statistics in real-time.
         * @param {Object} stats - Updated cache statistics
         * @private
         */
        _updateCacheStats: function(stats) {
            // Update cache statistics in real-time if cache dialog is open
            var oCacheDialog = this._dialogCache["cacheManager"];
            if (oCacheDialog && oCacheDialog.isOpen()) {
                var oModel = oCacheDialog.getModel("cache");
                if (oModel) {
                    var oData = oModel.getData();
                    oData.hitRates = stats.hitRates;
                    oModel.setData(oData);
                }
            }
        },

        /**
         * @function onConfirmCreateTask
         * @description Confirms and creates data management task with validation and security.
         * @public
         */
        onConfirmCreateTask: function() {
            var oDialog = this._dialogCache["createDataTask"];
            if (!oDialog) return;
            
            var oModel = oDialog.getModel("create");
            var oData = oModel.getData();
            
            // Validate required fields
            const validation = this._validateCreateTaskData(oData);
            if (!validation.isValid) {
                MessageBox.error(validation.message);
                return;
            }
            
            // Sanitize input data
            const sanitizedData = this._sanitizeCreateTaskData(oData);
            
            oDialog.setBusy(true);
            
            // Create secure AJAX configuration
            const ajaxConfig = this._securityUtils.createSecureAjaxConfig({
                url: "/a2a/agent8/v1/data-tasks",
                type: "POST",
                data: JSON.stringify(sanitizedData),
                success: function(data) {
                    oDialog.setBusy(false);
                    oDialog.close();
                    var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                    var sSuccessMsg = oBundle.getText("success.taskCreated") || "Data management task created successfully";
                    MessageToast.show(sSuccessMsg);
                    this._extensionAPI.refresh();
                    this._securityUtils.auditLog('DATA_TASK_CREATED', { taskName: sanitizedData.taskName });
                }.bind(this),
                error: function(xhr) {
                    oDialog.setBusy(false);
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                    var sErrorMsg = oBundle.getText("error.createTaskFailed") || "Failed to create task";
                    MessageBox.error(sErrorMsg + ": " + errorMsg);
                    this._securityUtils.auditLog('DATA_TASK_CREATE_FAILED', { error: errorMsg });
                }.bind(this)
            });
            
            jQuery.ajax(ajaxConfig);
        },

        /**
         * @function onCancelCreateTask
         * @description Cancels task creation and closes dialog.
         * @public
         */
        onCancelCreateTask: function() {
            var oDialog = this._dialogCache["createDataTask"];
            if (oDialog) {
                oDialog.close();
            }
        },
        
        /**
         * @function _startRealtimeUpdates
         * @description Establishes EventSource connection for real-time updates.
         * @private
         */
        _startRealtimeUpdates: function() {
            if (this._realtimeEventSource) {
                this._realtimeEventSource.close();
            }
            
            this._realtimeEventSource = new EventSource("/a2a/agent8/v1/realtime-updates");
            
            this._realtimeEventSource.onmessage = function(event) {
                var data = JSON.parse(event.data);
                
                if (data.type === "storage_alert") {
                    const safeMessage = this._securityUtils.encodeHTML(data.message || 'Storage alert occurred');
                    var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                    var sAlertMsg = oBundle.getText("msg.storageAlert") || "Storage Alert";
                    MessageToast.show(sAlertMsg + ": " + safeMessage);
                } else if (data.type === "cache_stats") {
                    this._updateCacheStats(data.stats);
                } else if (data.type === "operation_complete") {
                    const safeOperation = this._securityUtils.encodeHTML(data.operation || 'Unknown operation');
                    var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                    var sCompleteMsg = oBundle.getText("msg.operationComplete") || "Operation completed";
                    MessageToast.show(sCompleteMsg + ": " + safeOperation);
                    this._extensionAPI.refresh();
                }
            }.bind(this);
            
            this._realtimeEventSource.onerror = function() {
                var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                var sErrorMsg = oBundle.getText("error.realtimeDisconnected") || "Real-time updates disconnected";
                MessageToast.show(sErrorMsg);
            }.bind(this);
        },

        /**
         * @function _validateCreateTaskData
         * @description Validates create task data for security and business rules.
         * @param {Object} oData - Task data to validate
         * @returns {Object} Validation result with isValid flag and message
         * @private
         */
        _validateCreateTaskData: function(oData) {
            if (!oData.taskName || !oData.taskName.trim()) {
                return { isValid: false, message: "Task name is required" };
            }
            
            const taskNameValidation = this._securityUtils.validateInput(oData.taskName, 'text', {
                required: true,
                minLength: 3,
                maxLength: 100
            });
            if (!taskNameValidation.isValid) {
                return { isValid: false, message: "Task name: " + taskNameValidation.message };
            }
            
            if (!oData.datasetName || !oData.datasetName.trim()) {
                return { isValid: false, message: "Dataset name is required" };
            }
            
            const datasetValidation = this._securityUtils.validateInput(oData.datasetName, 'datasetName', {
                required: true
            });
            if (!datasetValidation.isValid) {
                return { isValid: false, message: "Dataset name: " + datasetValidation.message };
            }
            
            if (!oData.operationType || !oData.operationType.trim()) {
                return { isValid: false, message: "Operation type is required" };
            }
            
            return { isValid: true };
        },

        /**
         * @function _sanitizeCreateTaskData
         * @description Sanitizes create task data to prevent XSS and injection attacks.
         * @param {Object} oData - Task data to sanitize
         * @returns {Object} Sanitized task data
         * @private
         */
        _sanitizeCreateTaskData: function(oData) {
            return {
                taskName: this._securityUtils.sanitizeInput(oData.taskName),
                description: this._securityUtils.sanitizeInput(oData.description || ''),
                datasetName: this._securityUtils.sanitizeInput(oData.datasetName),
                operationType: this._securityUtils.sanitizeInput(oData.operationType),
                storageType: this._securityUtils.sanitizeInput(oData.storageType || 'HANA'),
                storageBackend: this._securityUtils.sanitizeInput(oData.storageBackend || ''),
                priority: this._securityUtils.sanitizeInput(oData.priority || 'MEDIUM'),
                compressionEnabled: Boolean(oData.compressionEnabled),
                encryptionEnabled: Boolean(oData.encryptionEnabled),
                cacheEnabled: Boolean(oData.cacheEnabled),
                versioningEnabled: Boolean(oData.versioningEnabled)
            };
        },

        /**
         * @function onCreateTransformation
         * @description Creates new data transformation task with comprehensive validation.
         * @public
         * @memberof a2a.network.agent8.ext.controller.ListReportExt
         * @since 1.0.0
         */
        onCreateTransformation: function() {
            if (!this._securityUtils.hasRole("TransformationManager")) {
                MessageBox.error("Access denied: Insufficient privileges for creating transformations");
                this._securityUtils.auditLog("CREATE_TRANSFORMATION_ACCESS_DENIED", { action: "create_transformation" });
                return;
            }
            
            this._getOrCreateDialog("createTransformation", "a2a.network.agent8.ext.fragment.CreateTransformation")
                .then(function(oDialog) {
                    var oModel = new JSONModel({
                        transformationName: "",
                        description: "",
                        sourceDataset: "",
                        targetDataset: "",
                        transformationType: "MAPPING",
                        transformationRules: [],
                        validationRules: [],
                        schedule: {
                            enabled: false,
                            frequency: "DAILY",
                            time: "00:00"
                        },
                        processingOptions: {
                            parallelProcessing: true,
                            batchSize: 1000,
                            errorHandling: "CONTINUE",
                            backupBeforeTransform: true
                        }
                    });
                    oDialog.setModel(oModel, "transformation");
                    oDialog.open();
                    
                    this._loadTransformationOptions(oDialog);
                    this._securityUtils.auditLog("CREATE_TRANSFORMATION_INITIATED", { action: "create_transformation" });
                }.bind(this));
        },

        /**
         * @function onScheduleTransformation
         * @description Schedules transformation tasks for selected items.
         * @public
         * @memberof a2a.network.agent8.ext.controller.ListReportExt
         * @since 1.0.0
         */
        onScheduleTransformation: function() {
            var oTable = this._extensionAPI.getTable();
            var aSelectedContexts = oTable.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageBox.warning("Please select at least one transformation task to schedule.");
                return;
            }
            
            if (!this._securityUtils.hasRole("TransformationManager")) {
                MessageBox.error("Access denied: Insufficient privileges for scheduling transformations");
                this._securityUtils.auditLog("SCHEDULE_TRANSFORMATION_ACCESS_DENIED", { action: "schedule_transformation" });
                return;
            }
            
            var aTaskNames = aSelectedContexts.map(function(oContext) {
                return this._securityUtils.sanitizeInput(oContext.getProperty("taskName"));
            }.bind(this));
            
            MessageBox.confirm(
                "Schedule transformation for " + aSelectedContexts.length + " selected task(s)?\\n\\n" +
                "Tasks: " + aTaskNames.join(", "),
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._openScheduleDialog(aSelectedContexts);
                        }
                    }.bind(this)
                }
            );
        },

        /**
         * @function onBatchTransformation
         * @description Initiates batch transformation for selected tasks.
         * @public
         * @memberof a2a.network.agent8.ext.controller.ListReportExt
         * @since 1.0.0
         */
        onBatchTransformation: function() {
            var oTable = this._extensionAPI.getTable();
            var aSelectedContexts = oTable.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageBox.warning("Please select at least one transformation task for batch processing.");
                return;
            }
            
            if (!this._securityUtils.hasRole("TransformationManager")) {
                MessageBox.error("Access denied: Insufficient privileges for batch transformations");
                this._securityUtils.auditLog("BATCH_TRANSFORMATION_ACCESS_DENIED", { action: "batch_transformation" });
                return;
            }
            
            var aTaskIds = aSelectedContexts.map(function(oContext) {
                return this._securityUtils.sanitizeInput(oContext.getProperty("ID"));
            }.bind(this));
            
            if (aTaskIds.length > 20) {
                MessageBox.error("Batch size limited to 20 tasks for security and performance reasons");
                return;
            }
            
            MessageBox.confirm(
                "Start batch transformation for " + aSelectedContexts.length + " selected task(s)?\\n\\n" +
                "This will execute all selected transformations in parallel with monitoring.",
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._executeBatchTransformation(aTaskIds);
                        }
                    }.bind(this)
                }
            );
        },

        /**
         * @function _loadTransformationOptions
         * @description Loads available transformation options and datasets.
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _loadTransformationOptions: function(oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["createTransformation"];
            if (!oTargetDialog) return;
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    const ajaxConfig = this._securityUtils.createSecureAjaxConfig({
                        url: "/a2a/agent8/v1/transformation-options",
                        type: "GET",
                        success: function(data) {
                            var oModel = oTargetDialog.getModel("transformation");
                            var oData = oModel.getData();
                            oData.availableDatasets = this._securityUtils.sanitizeArray(data.datasets || []);
                            oData.transformationTypes = this._securityUtils.sanitizeArray(data.types || []);
                            oData.ruleTemplates = this._securityUtils.sanitizeArray(data.templates || []);
                            oModel.setData(oData);
                            resolve(data);
                        }.bind(this),
                        error: function(xhr) {
                            const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                            reject(new Error("Failed to load transformation options: " + errorMsg));
                        }.bind(this)
                    });
                    jQuery.ajax(ajaxConfig);
                }.bind(this));
            }.bind(this)).catch(function(error) {
                MessageBox.error(error.message);
            });
        },

        /**
         * @function _openScheduleDialog
         * @description Opens scheduling dialog for selected transformations.
         * @param {Array} aContexts - Selected transformation contexts
         * @private
         */
        _openScheduleDialog: function(aContexts) {
            this._getOrCreateDialog("scheduleTransformation", "a2a.network.agent8.ext.fragment.ScheduleTransformation")
                .then(function(oDialog) {
                    var aTaskIds = aContexts.map(function(oContext) {
                        return this._securityUtils.sanitizeInput(oContext.getProperty("ID"));
                    }.bind(this));
                    
                    var oModel = new JSONModel({
                        selectedTasks: aTaskIds,
                        scheduleType: "IMMEDIATE",
                        scheduledDateTime: new Date(Date.now() + 60000), // 1 minute from now
                        recurrence: {
                            enabled: false,
                            frequency: "DAILY",
                            interval: 1,
                            endDate: null
                        },
                        executionOptions: {
                            parallelExecution: false,
                            maxConcurrency: 3,
                            continueOnError: true,
                            notifyOnCompletion: true
                        }
                    });
                    oDialog.setModel(oModel, "schedule");
                    oDialog.open();
                    
                    this._securityUtils.auditLog("SCHEDULE_DIALOG_OPENED", { taskCount: aTaskIds.length });
                }.bind(this));
        },

        /**
         * @function _executeBatchTransformation
         * @description Executes batch transformation with progress monitoring.
         * @param {Array} aTaskIds - Array of task IDs to transform
         * @private
         */
        _executeBatchTransformation: function(aTaskIds) {
            this.base.getView().setBusy(true);
            
            const requestData = {
                taskIds: aTaskIds,
                batchSize: Math.min(aTaskIds.length, 20),
                executionMode: "PARALLEL",
                continueOnError: true,
                generateReport: true
            };
            
            const ajaxConfig = this._securityUtils.createSecureAjaxConfig({
                url: "/a2a/agent8/v1/batch-transformation",
                type: "POST",
                data: JSON.stringify(requestData),
                success: function(data) {
                    this.base.getView().setBusy(false);
                    
                    MessageBox.success(
                        "Batch transformation initiated successfully!\\n" +
                        "Job ID: " + this._securityUtils.sanitizeInput(data.jobId) + "\\n" +
                        "Processing " + data.taskCount + " transformation(s)\\n" +
                        "Estimated time: " + this._securityUtils.sanitizeInput(data.estimatedTime) + " minutes"
                    );
                    
                    this._extensionAPI.refresh();
                    this._startBatchMonitoring(data.jobId);
                    
                    this._securityUtils.auditLog("BATCH_TRANSFORMATION_STARTED", {
                        jobId: data.jobId,
                        taskCount: data.taskCount
                    });
                }.bind(this),
                error: function(xhr) {
                    this.base.getView().setBusy(false);
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    MessageBox.error("Batch transformation failed: " + errorMsg);
                    this._securityUtils.auditLog("BATCH_TRANSFORMATION_FAILED", { error: errorMsg });
                }.bind(this)
            });
            
            jQuery.ajax(ajaxConfig);
        },

        /**
         * @function _startBatchMonitoring
         * @description Starts real-time monitoring of batch transformation progress.
         * @param {string} sJobId - Batch job ID to monitor
         * @private
         */
        _startBatchMonitoring: function(sJobId) {
            if (this._batchMonitoringEventSource) {
                this._batchMonitoringEventSource.close();
            }
            
            const streamUrl = "/a2a/agent8/v1/batch-transformation/" + encodeURIComponent(sJobId) + "/stream";
            
            if (!this._securityUtils.validateEventSourceUrl(streamUrl)) {
                MessageBox.error("Invalid batch monitoring stream URL");
                return;
            }
            
            this._batchMonitoringEventSource = new EventSource(streamUrl);
            
            this._batchMonitoringEventSource.onmessage = function(event) {
                try {
                    var data = JSON.parse(event.data);
                    
                    if (data.type === "progress") {
                        const progress = Math.max(0, Math.min(100, parseInt(data.progress) || 0));
                        const completed = parseInt(data.completed) || 0;
                        const total = parseInt(data.total) || 0;
                        MessageToast.show("Batch progress: " + completed + "/" + total + " (" + progress + "%)");
                    } else if (data.type === "task_completed") {
                        const taskName = this._securityUtils.sanitizeInput(data.taskName);
                        MessageToast.show("Completed: " + taskName);
                    } else if (data.type === "batch_completed") {
                        this._batchMonitoringEventSource.close();
                        MessageBox.success("Batch transformation completed successfully!");
                        this._extensionAPI.refresh();
                        this._securityUtils.auditLog("BATCH_TRANSFORMATION_COMPLETED", { jobId: sJobId });
                    } else if (data.type === "batch_failed") {
                        this._batchMonitoringEventSource.close();
                        const errorMsg = this._securityUtils.sanitizeInput(data.error || "Unknown error");
                        MessageBox.error("Batch transformation failed: " + errorMsg);
                        this._securityUtils.auditLog("BATCH_TRANSFORMATION_FAILED", { jobId: sJobId, error: errorMsg });
                    }
                } catch (e) {
                    this._batchMonitoringEventSource.close();
                    MessageBox.error("Invalid data received from batch monitoring");
                }
            }.bind(this);
            
            this._batchMonitoringEventSource.onerror = function() {
                if (this._batchMonitoringEventSource) {
                    this._batchMonitoringEventSource.close();
                    this._batchMonitoringEventSource = null;
                }
                MessageBox.error("Lost connection to batch monitoring");
            }.bind(this);
        },

        /**
         * @function onConfirmCreateTransformation
         * @description Confirms and creates new transformation with validation.
         * @public
         */
        onConfirmCreateTransformation: function() {
            var oDialog = this._dialogCache["createTransformation"];
            if (!oDialog) return;
            
            var oModel = oDialog.getModel("transformation");
            var oData = oModel.getData();
            
            // Validate transformation data
            const validation = this._validateTransformationData(oData);
            if (!validation.isValid) {
                MessageBox.error(validation.message);
                return;
            }
            
            oDialog.setBusy(true);
            
            const sanitizedData = this._sanitizeTransformationData(oData);
            
            const ajaxConfig = this._securityUtils.createSecureAjaxConfig({
                url: "/a2a/agent8/v1/transformations",
                type: "POST",
                data: JSON.stringify(sanitizedData),
                success: function(data) {
                    oDialog.setBusy(false);
                    oDialog.close();
                    MessageToast.show("Transformation created successfully");
                    this._extensionAPI.refresh();
                    
                    this._securityUtils.auditLog("TRANSFORMATION_CREATED", {
                        transformationName: sanitizedData.transformationName,
                        transformationId: data.id
                    });
                }.bind(this),
                error: function(xhr) {
                    oDialog.setBusy(false);
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    MessageBox.error("Failed to create transformation: " + errorMsg);
                    this._securityUtils.auditLog("TRANSFORMATION_CREATE_FAILED", { error: errorMsg });
                }.bind(this)
            });
            
            jQuery.ajax(ajaxConfig);
        },

        /**
         * @function onConfirmScheduleTransformation
         * @description Confirms and schedules selected transformations.
         * @public
         */
        onConfirmScheduleTransformation: function() {
            var oDialog = this._dialogCache["scheduleTransformation"];
            if (!oDialog) return;
            
            var oModel = oDialog.getModel("schedule");
            var oData = oModel.getData();
            
            if (oData.scheduleType === "SCHEDULED" && (!oData.scheduledDateTime || oData.scheduledDateTime <= new Date())) {
                MessageBox.error("Please select a valid future date and time for scheduling");
                return;
            }
            
            oDialog.setBusy(true);
            
            const sanitizedData = {
                taskIds: this._securityUtils.sanitizeArray(oData.selectedTasks),
                scheduleType: this._securityUtils.sanitizeInput(oData.scheduleType),
                scheduledDateTime: oData.scheduledDateTime ? oData.scheduledDateTime.toISOString() : null,
                recurrence: {
                    enabled: Boolean(oData.recurrence.enabled),
                    frequency: this._securityUtils.sanitizeInput(oData.recurrence.frequency || "DAILY"),
                    interval: Math.max(1, Math.min(365, parseInt(oData.recurrence.interval) || 1)),
                    endDate: oData.recurrence.endDate ? oData.recurrence.endDate.toISOString() : null
                },
                executionOptions: {
                    parallelExecution: Boolean(oData.executionOptions.parallelExecution),
                    maxConcurrency: Math.max(1, Math.min(10, parseInt(oData.executionOptions.maxConcurrency) || 3)),
                    continueOnError: Boolean(oData.executionOptions.continueOnError),
                    notifyOnCompletion: Boolean(oData.executionOptions.notifyOnCompletion)
                }
            };
            
            const ajaxConfig = this._securityUtils.createSecureAjaxConfig({
                url: "/a2a/agent8/v1/schedule-transformation",
                type: "POST",
                data: JSON.stringify(sanitizedData),
                success: function(data) {
                    oDialog.setBusy(false);
                    oDialog.close();
                    MessageBox.success(
                        "Transformation scheduled successfully!\\n" +
                        "Schedule ID: " + this._securityUtils.sanitizeInput(data.scheduleId)
                    );
                    this._extensionAPI.refresh();
                    
                    this._securityUtils.auditLog("TRANSFORMATION_SCHEDULED", {
                        scheduleId: data.scheduleId,
                        taskCount: sanitizedData.taskIds.length
                    });
                }.bind(this),
                error: function(xhr) {
                    oDialog.setBusy(false);
                    const errorMsg = this._securityUtils.sanitizeErrorMessage(xhr.responseText);
                    MessageBox.error("Failed to schedule transformation: " + errorMsg);
                    this._securityUtils.auditLog("TRANSFORMATION_SCHEDULE_FAILED", { error: errorMsg });
                }.bind(this)
            });
            
            jQuery.ajax(ajaxConfig);
        },

        /**
         * @function _validateTransformationData
         * @description Validates transformation creation data.
         * @param {Object} oData - Transformation data to validate
         * @returns {Object} Validation result
         * @private
         */
        _validateTransformationData: function(oData) {
            if (!oData.transformationName || !oData.transformationName.trim()) {
                return { isValid: false, message: "Transformation name is required" };
            }
            
            if (!oData.sourceDataset || !oData.sourceDataset.trim()) {
                return { isValid: false, message: "Source dataset is required" };
            }
            
            if (!oData.targetDataset || !oData.targetDataset.trim()) {
                return { isValid: false, message: "Target dataset is required" };
            }
            
            if (!oData.transformationType || !oData.transformationType.trim()) {
                return { isValid: false, message: "Transformation type is required" };
            }
            
            return { isValid: true };
        },

        /**
         * @function _sanitizeTransformationData
         * @description Sanitizes transformation data for security.
         * @param {Object} oData - Transformation data to sanitize
         * @returns {Object} Sanitized data
         * @private
         */
        _sanitizeTransformationData: function(oData) {
            return {
                transformationName: this._securityUtils.sanitizeInput(oData.transformationName),
                description: this._securityUtils.sanitizeInput(oData.description || ''),
                sourceDataset: this._securityUtils.sanitizeInput(oData.sourceDataset),
                targetDataset: this._securityUtils.sanitizeInput(oData.targetDataset),
                transformationType: this._securityUtils.sanitizeInput(oData.transformationType),
                transformationRules: this._securityUtils.sanitizeArray(oData.transformationRules || []),
                validationRules: this._securityUtils.sanitizeArray(oData.validationRules || []),
                schedule: {
                    enabled: Boolean(oData.schedule.enabled),
                    frequency: this._securityUtils.sanitizeInput(oData.schedule.frequency || "DAILY"),
                    time: this._securityUtils.sanitizeInput(oData.schedule.time || "00:00")
                },
                processingOptions: {
                    parallelProcessing: Boolean(oData.processingOptions.parallelProcessing),
                    batchSize: Math.max(100, Math.min(10000, parseInt(oData.processingOptions.batchSize) || 1000)),
                    errorHandling: this._securityUtils.sanitizeInput(oData.processingOptions.errorHandling || "CONTINUE"),
                    backupBeforeTransform: Boolean(oData.processingOptions.backupBeforeTransform)
                }
            };
        },

        /**
         * @function onCancelCreateTransformation
         * @description Cancels transformation creation dialog.
         * @public
         */
        onCancelCreateTransformation: function() {
            var oDialog = this._dialogCache["createTransformation"];
            if (oDialog) {
                oDialog.close();
            }
        },

        /**
         * @function onCancelScheduleTransformation
         * @description Cancels transformation scheduling dialog.
         * @public
         */
        onCancelScheduleTransformation: function() {
            var oDialog = this._dialogCache["scheduleTransformation"];
            if (oDialog) {
                oDialog.close();
            }
        }
    });
});