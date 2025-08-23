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
        }
    });
});