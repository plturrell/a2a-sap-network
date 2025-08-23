sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel",
    "a2a/network/agent12/ext/utils/SecurityUtils"
], function(ControllerExtension, MessageToast, MessageBox, Fragment, JSONModel, SecurityUtils) {
    "use strict";

    /**
     * @class a2a.network.agent12.ext.controller.ListReportExt
     * @extends sap.ui.core.mvc.ControllerExtension
     * @description Controller extension for Agent 12 List Report - Catalog Manager Agent.
     * Provides comprehensive service catalog management capabilities including resource discovery,
     * metadata management, registry synchronization, and search indexing with enterprise-grade security.
     */
    return ControllerExtension.extend("a2a.network.agent12.ext.controller.ListReportExt", {
        
        override: {
            /**
             * @function onInit
             * @description Initializes the controller extension with security utilities, device model, dialog caching, and real-time updates.
             * @override
             */
            onInit: function () {
                this._extensionAPI = this.base.getExtensionAPI();
                this._securityUtils = SecurityUtils;
                this._initializeDeviceModel();
                this._initializeDialogCache();
                this._initializePerformanceOptimizations();
                this._startRealtimeCatalogUpdates();
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
         * @description Performs search operation for catalog entries.
         * @param {string} sQuery - Search query
         * @private
         */
        _performSearch: function(sQuery) {
            // Implement search logic for catalog entries and resources
        },

        /**
         * @function onCatalogDashboard
         * @description Opens comprehensive catalog analytics dashboard with usage metrics and discovery statistics.
         * @public
         */
        onCatalogDashboard: function() {
            this._getOrCreateDialog("catalogDashboard", "a2a.network.agent12.ext.fragment.CatalogDashboard")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadDashboardData(oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Catalog Dashboard: " + error.message);
                });
        },

        /**
         * @function _getOrCreateDialog
         * @description Gets cached dialog or creates new one with accessibility and responsive features.
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
         * @function onCreateCatalogEntry
         * @description Opens dialog to create new catalog entry with resource metadata.
         * @public
         */
        onCreateCatalogEntry: function() {
            this._getOrCreateDialog("createCatalogEntry", "a2a.network.agent12.ext.fragment.CreateCatalogEntry")
                .then(function(oDialog) {
                    var oModel = new JSONModel({
                        resourceName: "",
                        description: "",
                        catalogType: "SERVICE",
                        resourceType: "REST_API",
                        category: "",
                        resourceUrl: "",
                        version: "1.0.0",
                        visibility: "PUBLIC",
                        authenticationMethod: "NONE",
                        keywords: "",
                        tags: "",
                        autoDiscovery: true,
                        searchable: true
                    });
                    oDialog.setModel(oModel, "create");
                    oDialog.open();
                    this._loadCatalogOptions(oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Create Catalog Entry dialog: " + error.message);
                });
        },

        /**
         * @function onServiceDiscovery
         * @description Opens service discovery interface for automatic resource detection.
         * @public
         */
        onServiceDiscovery: function() {
            this._getOrCreateDialog("serviceDiscovery", "a2a.network.agent12.ext.fragment.ServiceDiscovery")
                .then(function(oDialog) {
                    oDialog.open();
                    this._initializeDiscovery(oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Service Discovery: " + error.message);
                });
        },

        /**
         * @function onRegistryManager
         * @description Opens registry management interface for external catalog synchronization.
         * @public
         */
        onRegistryManager: function() {
            this._getOrCreateDialog("registryManager", "a2a.network.agent12.ext.fragment.RegistryManager")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadRegistryData(oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Registry Manager: " + error.message);
                });
        },

        /**
         * @function onCategoryManager
         * @description Opens category management interface for organizing catalog resources.
         * @public
         */
        onCategoryManager: function() {
            this._getOrCreateDialog("categoryManager", "a2a.network.agent12.ext.fragment.CategoryManager")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadCategoryData(oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Category Manager: " + error.message);
                });
        },

        /**
         * @function onSearchManager
         * @description Opens search index management interface for catalog search optimization.
         * @public
         */
        onSearchManager: function() {
            this._getOrCreateDialog("searchManager", "a2a.network.agent12.ext.fragment.SearchManager")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadSearchIndexData(oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Search Manager: " + error.message);
                });
        },

        /**
         * @function onMetadataEditor
         * @description Opens metadata editor for selected catalog entries.
         * @public
         */
        onMetadataEditor: function() {
            const oBinding = this.base.getView().byId("fe::table::CatalogEntries::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectEntriesFirst"));
                return;
            }

            this._getOrCreateDialog("metadataEditor", "a2a.network.agent12.ext.fragment.MetadataEditor")
                .then(function(oDialog) {
                    oDialog.open();
                    this._loadMetadataForEditing(aSelectedContexts[0], oDialog);
                }.bind(this))
                .catch(function(error) {
                    MessageBox.error("Failed to open Metadata Editor: " + error.message);
                });
        },

        /**
         * @function onDiscoveryScanner
         * @description Starts comprehensive resource discovery scan.
         * @public
         */
        onDiscoveryScanner: function() {
            MessageBox.confirm(
                this.getResourceBundle().getText("msg.startDiscoveryScan"),
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._startDiscoveryScan();
                        }
                    }.bind(this)
                }
            );
        },

        /**
         * @function onBulkValidation
         * @description Validates multiple catalog entries for consistency and compliance.
         * @public
         */
        onBulkValidation: function() {
            const oBinding = this.base.getView().byId("fe::table::CatalogEntries::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectEntriesFirst"));
                return;
            }

            this._validateBulkEntries(aSelectedContexts);
        },

        /**
         * @function onBulkPublishing
         * @description Publishes multiple catalog entries to make them available for discovery.
         * @public
         */
        onBulkPublishing: function() {
            const oBinding = this.base.getView().byId("fe::table::CatalogEntries::LineItem").getBinding("rows");
            const aSelectedContexts = oBinding.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageToast.show(this.getResourceBundle().getText("msg.selectEntriesFirst"));
                return;
            }

            MessageBox.confirm(
                this.getResourceBundle().getText("msg.publishEntriesConfirm", [aSelectedContexts.length]),
                {
                    onClose: function(oAction) {
                        if (oAction === MessageBox.Action.OK) {
                            this._publishBulkEntries(aSelectedContexts);
                        }
                    }.bind(this)
                }
            );
        },

        /**
         * @function _startRealtimeCatalogUpdates
         * @description Starts real-time updates for catalog changes and discovery events.
         * @private
         */
        _startRealtimeCatalogUpdates: function() {
            this._initializeWebSocket();
        },

        /**
         * @function _initializeWebSocket
         * @description Initializes secure WebSocket connection for real-time catalog updates.
         * @private
         */
        _initializeWebSocket: function() {
            if (this._ws) return;

            // Validate WebSocket URL for security
            if (!this._securityUtils.validateWebSocketUrl('wss://localhost:8012/catalog/updates')) {
                MessageBox.error("Invalid WebSocket URL");
                return;
            }

            try {
                this._ws = SecurityUtils.createSecureWebSocket('wss://localhost:8012/catalog/updates', {
                    onmessage: function(event) {
                        try {
                            const data = JSON.parse(event.data);
                            this._handleCatalogUpdate(data);
                        } catch (error) {
                            console.error("Error parsing WebSocket message:", error);
                        }
                    }.bind(this),
                    onerror: function(error) {
                        console.warn("Secure WebSocket error:", error);
                        this._initializePolling();
                    }.bind(this)
                });
                
                if (this._ws) {
                    this._ws.onclose = function() {
                        var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                        var sMessage = oBundle.getText("msg.websocketDisconnected") || "Connection lost. Reconnecting...";
                        MessageToast.show(sMessage);
                        setTimeout(() => this._initializeWebSocket(), 5000);
                    }.bind(this);
                }

            } catch (error) {
                console.warn("WebSocket connection failed, falling back to polling");
                this._initializePolling();
            }
        },

        /**
         * @function _initializePolling
         * @description Initializes polling fallback for real-time updates.
         * @private
         */
        _initializePolling: function() {
            this._pollInterval = setInterval(() => {
                this._refreshCatalogData();
            }, 10000);
        },

        /**
         * @function _handleCatalogUpdate
         * @description Handles real-time catalog updates from WebSocket.
         * @param {Object} data - Update data
         * @private
         */
        _handleCatalogUpdate: function(data) {
            try {
                // Sanitize incoming data
                const sanitizedData = SecurityUtils.sanitizeCatalogData(JSON.stringify(data));
                const parsedData = JSON.parse(sanitizedData);
                
                var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                
                switch (parsedData.type) {
                    case 'RESOURCE_REGISTERED':
                        var sRegisteredMsg = oBundle.getText("msg.resourceRegistered") || "Resource registered";
                        MessageToast.show(sRegisteredMsg);
                        this._refreshCatalogData();
                        break;
                    case 'DISCOVERY_COMPLETED':
                        var sDiscoveryMsg = oBundle.getText("msg.discoveryCompleted") || "Discovery completed";
                        MessageToast.show(sDiscoveryMsg);
                        this._refreshCatalogData();
                        break;
                    case 'INDEX_UPDATED':
                        var sIndexMsg = oBundle.getText("msg.searchIndexUpdated") || "Search index updated";
                        MessageToast.show(sIndexMsg);
                        break;
                    case 'REGISTRY_SYNCED':
                        var sSyncMsg = oBundle.getText("msg.registrySynced") || "Registry synchronized";
                        MessageToast.show(sSyncMsg);
                        this._refreshCatalogData();
                        break;
                    case 'METADATA_UPDATED':
                        this._refreshCatalogData();
                        break;
                }
            } catch (error) {
                console.error("Error processing catalog update:", error);
            }
        },

        /**
         * @function _loadDashboardData
         * @description Loads catalog dashboard data with statistics and growth metrics.
         * @param {sap.m.Dialog} oDialog - Target dialog (optional)
         * @private
         */
        _loadDashboardData: function(oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["catalogDashboard"];
            if (!oTargetDialog) return;
            
            oTargetDialog.setBusy(true);
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    var oModel = this.base.getView().getModel();
                    SecurityUtils.secureCallFunction(oModel, "/GetCatalogStatistics", {
                        success: function(data) {
                            resolve(data);
                        },
                        error: function(error) {
                            var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                            var sErrorMsg = oBundle.getText("error.loadingStatistics") || "Error loading statistics";
                            reject(new Error(sErrorMsg));
                        }.bind(this)
                    });
                }.bind(this));
            }.bind(this)).then(function(data) {
                oTargetDialog.setBusy(false);
                this._updateDashboardCharts(data, oTargetDialog);
            }.bind(this)).catch(function(error) {
                oTargetDialog.setBusy(false);
                MessageBox.error(error.message);
            });
        },

        /**
         * @function _initializeDiscovery
         * @description Initializes service discovery options and methods.
         * @param {sap.m.Dialog} oDialog - Target dialog (optional)
         * @private
         */
        _initializeDiscovery: function(oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["serviceDiscovery"];
            if (!oTargetDialog) return;
            
            oTargetDialog.setBusy(true);
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    var oModel = this.base.getView().getModel();
                    SecurityUtils.secureCallFunction(oModel, "/GetDiscoveryMethods", {
                        success: function(data) {
                            resolve(data);
                        },
                        error: function(error) {
                            var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                            var sErrorMsg = oBundle.getText("error.loadingDiscoveryMethods") || "Error loading discovery methods";
                            reject(new Error(sErrorMsg));
                        }.bind(this)
                    });
                }.bind(this));
            }.bind(this)).then(function(data) {
                oTargetDialog.setBusy(false);
                this._updateDiscoveryOptions(data, oTargetDialog);
            }.bind(this)).catch(function(error) {
                oTargetDialog.setBusy(false);
                MessageBox.error(error.message);
            });
        },

        /**
         * @function _loadRegistryData
         * @description Loads registry configurations and synchronization status.
         * @param {sap.m.Dialog} oDialog - Target dialog (optional)
         * @private
         */
        _loadRegistryData: function(oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["registryManager"];
            if (!oTargetDialog) return;
            
            oTargetDialog.setBusy(true);
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    var oModel = this.base.getView().getModel();
                    SecurityUtils.secureCallFunction(oModel, "/GetRegistryConfigurations", {
                        success: function(data) {
                            resolve(data);
                        },
                        error: function(error) {
                            var oBundle = this.base.getView().getModel("i18n").getResourceBundle();
                            var sErrorMsg = oBundle.getText("error.loadingRegistryData") || "Error loading registry data";
                            reject(new Error(sErrorMsg));
                        }.bind(this)
                    });
                }.bind(this));
            }.bind(this)).then(function(data) {
                oTargetDialog.setBusy(false);
                this._updateRegistryList(data, oTargetDialog);
            }.bind(this)).catch(function(error) {
                oTargetDialog.setBusy(false);
                MessageBox.error(error.message);
            });
        },

        _loadCategoryData: function() {
            const oModel = this.getView().getModel();
            
            SecurityUtils.secureCallFunction(oModel, "/GetServiceCategories", {
                success: function(data) {
                    this._updateCategoryTree(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingCategoryData"));
                }.bind(this)
            });
        },

        _loadSearchIndexData: function() {
            const oModel = this.getView().getModel();
            
            SecurityUtils.secureCallFunction(oModel, "/GetSearchIndexes", {
                success: function(data) {
                    this._updateSearchIndexList(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingSearchIndexData"));
                }.bind(this)
            });
        },

        _loadMetadataForEditing: function(oContext) {
            const oModel = this.getView().getModel();
            const sEntryId = oContext.getObject().entryId;
            
            SecurityUtils.secureCallFunction(oModel, "/GetMetadataProperties", {
                urlParameters: {
                    entryId: sEntryId
                },
                success: function(data) {
                    this._displayMetadataEditor(data);
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.loadingMetadata"));
                }.bind(this)
            });
        },

        _startDiscoveryScan: function() {
            const oModel = this.getView().getModel();
            
            MessageToast.show(this.getResourceBundle().getText("msg.discoveryScanStarted"));
            
            if (!SecurityUtils.checkCatalogAuth('StartResourceDiscovery', {})) {
                return;
            }

            SecurityUtils.secureCallFunction(oModel, "/StartResourceDiscovery", {
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.discoveryCompleted"));
                    this._refreshCatalogData();
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.discoveryFailed"));
                }.bind(this)
            });
        },

        _validateBulkEntries: function(aSelectedContexts) {
            const oModel = this.getView().getModel();
            const aEntryIds = aSelectedContexts.map(ctx => ctx.getObject().entryId);
            
            MessageToast.show(this.getResourceBundle().getText("msg.bulkValidationStarted", [aEntryIds.length]));
            
            if (!SecurityUtils.checkCatalogAuth('ValidateCatalogEntries', {})) {
                return;
            }

            SecurityUtils.secureCallFunction(oModel, "/ValidateCatalogEntries", {
                urlParameters: {
                    entryIds: aEntryIds.join(',')
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.bulkValidationCompleted"));
                    this._refreshCatalogData();
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.validationFailed"));
                }.bind(this)
            });
        },

        _publishBulkEntries: function(aSelectedContexts) {
            const oModel = this.getView().getModel();
            const aEntryIds = aSelectedContexts.map(ctx => ctx.getObject().entryId);
            
            MessageToast.show(this.getResourceBundle().getText("msg.bulkPublishingStarted", [aEntryIds.length]));
            
            if (!SecurityUtils.checkCatalogAuth('PublishCatalogEntries', {})) {
                return;
            }

            SecurityUtils.secureCallFunction(oModel, "/PublishCatalogEntries", {
                urlParameters: {
                    entryIds: aEntryIds.join(',')
                },
                success: function(data) {
                    MessageToast.show(this.getResourceBundle().getText("msg.bulkPublishingCompleted"));
                    this._refreshCatalogData();
                }.bind(this),
                error: function(error) {
                    MessageToast.show(this.getResourceBundle().getText("error.publishingFailed"));
                }.bind(this)
            });
        },

        _refreshCatalogData: function() {
            const oBinding = this.base.getView().byId("fe::table::CatalogEntries::LineItem").getBinding("rows");
            oBinding.refresh();
        },

        /**
         * @function _loadCatalogOptions
         * @description Loads catalog configuration options for entry creation.
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _loadCatalogOptions: function(oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["createCatalogEntry"];
            if (!oTargetDialog) return;
            
            this._withErrorRecovery(function() {
                return new Promise(function(resolve, reject) {
                    var oModel = this.base.getView().getModel();
                    SecurityUtils.secureCallFunction(oModel, "/GetCatalogOptions", {
                        success: function(data) {
                            var oCreateModel = oTargetDialog.getModel("create");
                            var oCreateData = oCreateModel.getData();
                            oCreateData.availableTypes = data.catalogTypes;
                            oCreateData.resourceTypes = data.resourceTypes;
                            oCreateData.categories = data.categories;
                            oCreateData.authMethods = data.authenticationMethods;
                            oCreateModel.setData(oCreateData);
                            resolve(data);
                        },
                        error: function(error) {
                            reject(new Error("Failed to load catalog options"));
                        }
                    });
                }.bind(this));
            }.bind(this)).catch(function(error) {
                MessageBox.error(error.message);
            });
        },

        /**
         * @function _updateDashboardCharts
         * @description Updates catalog dashboard charts with growth and distribution metrics.
         * @param {Object} data - Dashboard data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _updateDashboardCharts: function(data, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["catalogDashboard"];
            if (!oTargetDialog) return;
            
            this._createCatalogGrowthChart(data.catalogGrowth, oTargetDialog);
            this._createResourceTypeChart(data.resourceTypes, oTargetDialog);
            this._createCategoryDistributionChart(data.categoryDistribution, oTargetDialog);
        },

        /**
         * @function _updateDiscoveryOptions
         * @description Updates service discovery options and configurations.
         * @param {Object} data - Discovery options data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _updateDiscoveryOptions: function(data, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["serviceDiscovery"];
            if (!oTargetDialog) return;
            
            var oDiscoveryModel = new JSONModel({
                methods: data.discoveryMethods,
                scanTargets: data.scanTargets,
                configurations: data.configurations,
                scheduledScans: data.scheduledScans
            });
            oTargetDialog.setModel(oDiscoveryModel, "discovery");
        },

        /**
         * @function _updateRegistryList
         * @description Updates registry configuration list and status.
         * @param {Object} data - Registry data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _updateRegistryList: function(data, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["registryManager"];
            if (!oTargetDialog) return;
            
            var oRegistryModel = new JSONModel({
                registries: data.registries,
                syncStatus: data.syncStatus,
                configurations: data.configurations,
                healthStatus: data.healthStatus
            });
            oTargetDialog.setModel(oRegistryModel, "registry");
        },

        /**
         * @function _updateCategoryTree
         * @description Updates category tree structure for catalog organization.
         * @param {Object} data - Category tree data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _updateCategoryTree: function(data, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["categoryManager"];
            if (!oTargetDialog) return;
            
            var oCategoryModel = new JSONModel({
                categoryTree: data.categories,
                statistics: data.categoryStats,
                serviceCount: data.serviceCount,
                hierarchy: data.hierarchy
            });
            oTargetDialog.setModel(oCategoryModel, "category");
        },

        /**
         * @function _updateSearchIndexList
         * @description Updates search index configurations and performance metrics.
         * @param {Object} data - Search index data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _updateSearchIndexList: function(data, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["searchManager"];
            if (!oTargetDialog) return;
            
            var oSearchModel = new JSONModel({
                indexes: data.searchIndexes,
                performance: data.performance,
                statistics: data.statistics,
                configuration: data.configuration
            });
            oTargetDialog.setModel(oSearchModel, "search");
        },

        /**
         * @function _displayMetadataEditor
         * @description Displays metadata properties in editor for modification.
         * @param {Object} data - Metadata data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _displayMetadataEditor: function(data, oDialog) {
            var oTargetDialog = oDialog || this._dialogCache["metadataEditor"];
            if (!oTargetDialog) return;
            
            var oMetadataModel = new JSONModel({
                properties: data.properties,
                schema: data.schema,
                validation: data.validation,
                history: data.changeHistory
            });
            oTargetDialog.setModel(oMetadataModel, "metadata");
        },

        /**
         * @function _createCatalogGrowthChart
         * @description Creates catalog growth chart for dashboard visualization.
         * @param {Object} data - Chart data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createCatalogGrowthChart: function(data, oDialog) {
            var oChartContainer = oDialog.byId("catalogGrowthChart");
            if (!oChartContainer || !data.catalogGrowth) return;
            
            var oChartModel = new JSONModel({
                chartData: data.catalogGrowth,
                config: {
                    title: this.getResourceBundle().getText("chart.catalogGrowth"),
                    xAxisLabel: this.getResourceBundle().getText("chart.time"),
                    yAxisLabel: this.getResourceBundle().getText("field.entryCount")
                }
            });
            oChartContainer.setModel(oChartModel, "chart");
        },
        
        /**
         * @function _createResourceTypeChart
         * @description Creates resource type distribution chart for dashboard.
         * @param {Object} data - Chart data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createResourceTypeChart: function(data, oDialog) {
            var oChartContainer = oDialog.byId("resourceTypeChart");
            if (!oChartContainer || !data.resourceTypes) return;
            
            var oChartModel = new JSONModel({
                chartData: data.resourceTypes,
                config: {
                    title: this.getResourceBundle().getText("chart.resourceTypes"),
                    showLegend: true,
                    colorPalette: ["#5cbae6", "#b6d7a8", "#ffd93d", "#ff7b7b", "#c5a5d0"]
                }
            });
            oChartContainer.setModel(oChartModel, "chart");
        },
        
        /**
         * @function _createCategoryDistributionChart
         * @description Creates category distribution chart for analytics dashboard.
         * @param {Object} data - Chart data
         * @param {sap.m.Dialog} oDialog - Target dialog
         * @private
         */
        _createCategoryDistributionChart: function(data, oDialog) {
            var oChartContainer = oDialog.byId("categoryDistributionChart");
            if (!oChartContainer || !data.categoryDistribution) return;
            
            var oChartModel = new JSONModel({
                chartData: data.categoryDistribution,
                config: {
                    title: this.getResourceBundle().getText("chart.categoryDistribution"),
                    showDataLabels: true,
                    enableDrillDown: true
                }
            });
            oChartContainer.setModel(oChartModel, "chart");
        },
        
        /**
         * @function _cleanupResources
         * @description Cleans up resources and connections on controller destruction.
         * @private
         */
        _cleanupResources: function() {
            // Close WebSocket connection
            if (this._ws) {
                this._ws.close();
                this._ws = null;
            }
            
            // Clear polling interval
            if (this._pollInterval) {
                clearInterval(this._pollInterval);
                this._pollInterval = null;
            }
            
            // Clear cached dialogs
            for (var sDialogId in this._dialogCache) {
                var oDialog = this._dialogCache[sDialogId];
                if (oDialog && oDialog.destroy) {
                    oDialog.destroy();
                }
            }
            this._dialogCache = {};
            
            // Clear throttled and debounced functions
            if (this._throttledDashboardUpdate) {
                this._throttledDashboardUpdate = null;
            }
            if (this._debouncedSearch) {
                this._debouncedSearch = null;
            }
        },

        /**
         * @function getResourceBundle
         * @description Gets the i18n resource bundle for text translations.
         * @returns {sap.ui.model.resource.ResourceModel} Resource bundle
         * @public
         */
        getResourceBundle: function() {
            return this.base.getView().getModel("i18n").getResourceBundle();
        }
    });
});