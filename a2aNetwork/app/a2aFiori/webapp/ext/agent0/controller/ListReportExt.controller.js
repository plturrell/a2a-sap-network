sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel",
    "a2a/network/agent0/ext/utils/SecurityUtils"
], function (ControllerExtension, MessageToast, MessageBox, Fragment, JSONModel, SecurityUtils) {
    'use strict';

    /**
     * @class a2a.network.agent0.ext.controller.ListReportExt
     * @extends sap.ui.core.mvc.ControllerExtension
     * @description Controller extension for Agent 0 List Report - Data Product Agent.
     * Provides comprehensive data product management capabilities with Dublin Core metadata,
     * schema registry integration, quality assessment, and enterprise-grade security.
     */
    return ControllerExtension.extend("a2a.network.agent0.ext.controller.ListReportExt", {
        
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
                this._startRealtimeProductUpdates();
                this._initializeModels();
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
         * @function _getOrCreateDialog
         * @description Gets existing dialog from cache or creates new one with accessibility and responsive features.
         * @param {string} sDialogId - Dialog identifier
         * @param {string} sFragmentName - Fragment name
         * @returns {Promise<sap.m.Dialog>} Dialog instance
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
                that._enableDialogAccessibility(oDialog);
                that._optimizeDialogForDevice(oDialog);
                return oDialog;
            });
        },
        
        /**
         * @function _withErrorRecovery
         * @description Wraps operation with error recovery and exponential backoff.
         * @param {Function} fnOperation - Operation to execute
         * @param {string} sOperationName - Operation name for logging
         * @returns {Promise} Promise that resolves with operation result
         * @private
         */
        _withErrorRecovery: function(fnOperation, sOperationName) {
            var that = this;
            var oConfig = this._errorRecoveryConfig;
            
            function attempt(nRetry) {
                return fnOperation().catch(function(oError) {
                    if (nRetry < oConfig.maxRetries) {
                        var nDelay = oConfig.exponentialBackoff ? 
                            oConfig.retryDelay * Math.pow(2, nRetry) : oConfig.retryDelay;
                        
                        MessageToast.show(that._getResourceBundle().getText("recovery.retrying"));
                        
                        return new Promise(function(resolve) {
                            setTimeout(function() {
                                resolve(attempt(nRetry + 1));
                            }, nDelay);
                        });
                    } else {
                        MessageToast.show(that._getResourceBundle().getText("recovery.maxRetriesReached"));
                        throw oError;
                    }
                });
            }
            
            return attempt(0);
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
                    setTimeout(function() {
                        inThrottle = false;
                    }, limit);
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
                var args = arguments;
                var context = this;
                clearTimeout(timeoutId);
                timeoutId = setTimeout(function() {
                    fn.apply(context, args);
                }, delay);
            };
        },
        
        /**
         * @function _enableDialogAccessibility
         * @description Enables accessibility features for dialogs.
         * @param {sap.m.Dialog} oDialog - Dialog to enhance
         * @private
         */
        _enableDialogAccessibility: function(oDialog) {
            // Add ARIA attributes
            oDialog.addStyleClass("a2aAccessibleDialog");
            
            // Enable keyboard navigation
            oDialog.attachAfterOpen(function() {
                var oFirstInput = oDialog.$().find('input, button, select, textarea').first();
                if (oFirstInput.length) {
                    oFirstInput.focus();
                }
            });
            
            // Add escape key handler
            oDialog.addEventDelegate({
                onkeydown: function(oEvent) {
                    if (oEvent.keyCode === 27) { // Escape key
                        oDialog.close();
                    }
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
                oDialog.addStyleClass("a2aMobileOptimized");
            } else if (sap.ui.Device.system.tablet) {
                oDialog.setContentWidth("80%");
                oDialog.addStyleClass("a2aTabletOptimized");
            } else {
                oDialog.setContentWidth("60%");
                oDialog.addStyleClass("a2aDesktopOptimized");
            }
        },
        
        /**
         * @function _startRealtimeProductUpdates
         * @description Initializes real-time updates via WebSocket with polling fallback.
         * @private
         */
        _startRealtimeProductUpdates: function() {
            if (this._ws) return;
            
            try {
                this._ws = SecurityUtils.createSecureWebSocket('wss://localhost:8000/dataproduct/updates', {
                    onmessage: function(event) {
                        var oData = JSON.parse(event.data);
                        this._handleProductUpdate(oData);
                    }.bind(this),
                    onerror: function(error) {
                        console.warn("Secure WebSocket error:", error);
                        this._initializePolling();
                    }.bind(this)
                });
                
                if (this._ws) {
                    this._ws.onclose = function() {
                        setTimeout(() => this._startRealtimeProductUpdates(), 5000);
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
                this._refreshProductData();
            }, 3000);
        },
        
        /**
         * @function _handleProductUpdate
         * @description Handles real-time product updates.
         * @param {object} oData - Update data
         * @private
         */
        _handleProductUpdate: function(oData) {
            switch (oData.type) {
                case 'PRODUCT_CREATED':
                    MessageToast.show(this._getResourceBundle().getText("msg.productCreated"));
                    this._refreshProductData();
                    break;
                case 'METADATA_UPDATED':
                    MessageToast.show(this._getResourceBundle().getText("msg.metadataUpdated"));
                    this._refreshProductData();
                    break;
                case 'VALIDATION_COMPLETED':
                    this._updateValidationStatus(oData);
                    break;
                case 'QUALITY_ASSESSED':
                    this._updateQualityMetrics(oData);
                    break;
            }
        },
        
        /**
         * @function _refreshProductData
         * @description Refreshes the product data in the table.
         * @private
         */
        _refreshProductData: function() {
            var oTable = this.base.getView().byId("fe::table::DataProducts::LineItem");
            if (oTable && oTable.getBinding("rows")) {
                oTable.getBinding("rows").refresh();
            }
        },
        
        /**
         * @function _updateValidationStatus
         * @description Updates validation status in real-time.
         * @param {object} oData - Validation data
         * @private
         */
        _updateValidationStatus: function(oData) {
            // Implementation for real-time validation status updates
        },
        
        /**
         * @function _updateQualityMetrics
         * @description Updates quality metrics in real-time.
         * @param {object} oData - Quality metrics data
         * @private
         */
        _updateQualityMetrics: function(oData) {
            // Implementation for real-time quality metrics updates
        },
        
        /**
         * @function _cleanupResources
         * @description Cleans up WebSocket connections and intervals.
         * @private
         */
        _cleanupResources: function() {
            if (this._ws) {
                this._ws.close();
                this._ws = null;
            }
            if (this._pollInterval) {
                clearInterval(this._pollInterval);
                this._pollInterval = null;
            }
        },
        
        /**
         * @function _performSearch
         * @description Performs debounced search operation.
         * @param {string} sQuery - Search query
         * @private
         */
        _performSearch: function(sQuery) {
            // Implementation for search functionality
        },
        
        /**
         * @function _initializeModels
         * @description Initialize models for the view with comprehensive data structure.
         * @private
         */
        _initializeModels: function () {
            const oViewModel = new JSONModel({
                busy: false,
                dashboardData: {
                    totalProducts: 0,
                    activeProducts: 0,
                    averageQuality: 0,
                    recentActivity: []
                }
            });
            this.base.getView().setModel(oViewModel, "viewModel");
        },

        /**
         * @function onCreateDataProduct
         * @description Handler for Create Data Product action with dialog caching and accessibility.
         */
        onCreateDataProduct: function () {
            this._withErrorRecovery(function() {
                return this._getOrCreateDialog("createDataProduct", "a2a.network.agent0.ext.fragment.CreateDataProduct")
                    .then(function(oDialog) {
                        oDialog.open();
                    });
            }.bind(this), "CreateDataProduct").catch(function(oError) {
                MessageToast.show(this._getResourceBundle().getText("error.dialogLoadFailed"));
            }.bind(this));
        },

        /**
         * @function onImportMetadata
         * @description Handler for Import Metadata action with error recovery and caching.
         */
        onImportMetadata: function () {
            this._withErrorRecovery(function() {
                return this._getOrCreateDialog("importMetadata", "a2a.network.agent0.ext.fragment.ImportMetadata")
                    .then(function(oDialog) {
                        oDialog.open();
                    });
            }.bind(this), "ImportMetadata").catch(function(oError) {
                MessageToast.show(this._getResourceBundle().getText("error.dialogLoadFailed"));
            }.bind(this));
        },

        /**
         * @function onOpenDashboard
         * @description Handler for Dashboard action with throttled data loading.
         */
        onOpenDashboard: function () {
            this._withErrorRecovery(function() {
                return this._getOrCreateDialog("dashboard", "a2a.network.agent0.ext.fragment.DataProductDashboard")
                    .then(function(oDialog) {
                        this._throttledDashboardUpdate();
                        oDialog.open();
                    }.bind(this));
            }.bind(this), "OpenDashboard").catch(function(oError) {
                MessageToast.show(this._getResourceBundle().getText("error.dashboardLoadFailed"));
            }.bind(this));
        },

        /**
         * @function onValidateMetadata
         * @description Handler for Validate Metadata action with comprehensive error handling.
         * @param {sap.ui.base.Event} oEvent - Event object
         */
        onValidateMetadata: function (oEvent) {
            var oTable = this.base.getView().byId("fe::table::DataProducts::LineItem");
            var aSelectedContexts = oTable.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageToast.show(this._getResourceBundle().getText("msg.selectDataProduct"));
                return;
            }
            
            this._withErrorRecovery(function() {
                return this._validateSelectedProducts(aSelectedContexts);
            }.bind(this), "ValidateMetadata").catch(function(oError) {
                MessageToast.show(this._getResourceBundle().getText("error.validationFailed"));
            }.bind(this));
        },

        /**
         * @function onBulkUpdate
         * @description Handler for Bulk Update action with selection validation and dialog caching.
         */
        onBulkUpdate: function () {
            var oTable = this.base.getView().byId("fe::table::DataProducts::LineItem");
            var aSelectedContexts = oTable.getSelectedContexts();
            
            if (aSelectedContexts.length === 0) {
                MessageToast.show(this._getResourceBundle().getText("msg.selectProductsToUpdate"));
                return;
            }
            
            this._withErrorRecovery(function() {
                return this._getOrCreateDialog("bulkUpdate", "a2a.network.agent0.ext.fragment.BulkUpdate")
                    .then(function(oDialog) {
                        oDialog.open();
                    });
            }.bind(this), "BulkUpdate").catch(function(oError) {
                MessageToast.show(this._getResourceBundle().getText("error.bulkUpdateFailed"));
            }.bind(this));
        },

        /**
         * @function onExportCatalog
         * @description Handler for Export Catalog action with format selection dialog.
         */
        onExportCatalog: function () {
            this._withErrorRecovery(function() {
                var oResourceBundle = this._getResourceBundle();
                var oDialog = new sap.m.Dialog({
                    title: oResourceBundle.getText("dialog.exportCatalog.title"),
                    type: "Message",
                    content: [
                        new sap.m.Text({ text: oResourceBundle.getText("msg.selectExportFormat") }),
                        new sap.m.RadioButtonGroup({
                            columns: 1,
                            selectedIndex: 0,
                            buttons: [
                                new sap.m.RadioButton({ text: oResourceBundle.getText("format.json") }),
                                new sap.m.RadioButton({ text: oResourceBundle.getText("format.csv") }),
                                new sap.m.RadioButton({ text: oResourceBundle.getText("format.xml") })
                            ]
                        })
                    ],
                    beginButton: new sap.m.Button({
                        text: oResourceBundle.getText("btn.export"),
                        type: "Emphasized",
                        press: function () {
                            var iSelectedIndex = oDialog.getContent()[1].getSelectedIndex();
                            var aFormats = ["json", "csv", "xml"];
                            this._exportCatalog(aFormats[iSelectedIndex]);
                            oDialog.close();
                        }.bind(this)
                    }),
                    endButton: new sap.m.Button({
                        text: oResourceBundle.getText("btn.cancel"),
                        press: function () {
                            oDialog.close();
                        }
                    }),
                    afterClose: function () {
                        oDialog.destroy();
                    }
                });
                
                this._enableDialogAccessibility(oDialog);
                this._optimizeDialogForDevice(oDialog);
                oDialog.open();
                return Promise.resolve();
            }.bind(this), "ExportCatalog").catch(function(oError) {
                MessageToast.show(this._getResourceBundle().getText("error.exportDialogFailed"));
            }.bind(this));
        },

        /**
         * @function _loadDashboardData
         * @description Load dashboard data with error recovery and security validation.
         * @returns {Promise} Promise that resolves when dashboard data is loaded
         * @private
         */
        _loadDashboardData: function () {
            var oViewModel = this.base.getView().getModel("viewModel");
            oViewModel.setProperty("/busy", true);
            
            return this._withErrorRecovery(function() {
                var oModel = this.base.getView().getModel();
                return SecurityUtils.secureCallFunction(oModel, "/GetDashboardMetrics", {
                    success: function(oData) {
                        oViewModel.setProperty("/dashboardData", oData);
                    }.bind(this),
                    error: function(oError) {
                        throw new Error("Dashboard data loading failed");
                    }
                });
            }.bind(this), "LoadDashboardData").catch(function(oError) {
                MessageToast.show(this._getResourceBundle().getText("error.loadingDashboardData"));
            }.bind(this)).finally(function() {
                oViewModel.setProperty("/busy", false);
            });
        },

        /**
         * @function _validateSelectedProducts
         * @description Validate selected products with comprehensive security checks.
         * @param {Array} aContexts - Array of selected contexts
         * @returns {Promise} Promise that resolves when validation is complete
         * @private
         */
        _validateSelectedProducts: function (aContexts) {
            var oModel = this.base.getView().getModel();
            var iSuccessCount = 0;
            var iFailureCount = 0;
            var aPromises = [];
            
            aContexts.forEach(function(oContext) {
                var oPromise = new Promise(function(resolve, reject) {
                    try {
                        var sPath = oContext.getPath();
                        var oData = oContext.getObject();
                        var validation = SecurityUtils.validateDublinCore(oData.dublinCore);
                        
                        if (!validation.valid) {
                            throw new Error(validation.error);
                        }
                        
                        SecurityUtils.secureCallFunction(oModel, sPath + "/validateMetadata", {
                            success: function() {
                                iSuccessCount++;
                                resolve();
                            },
                            error: function() {
                                iFailureCount++;
                                resolve(); // Don't reject, just count failures
                            }
                        });
                    } catch (error) {
                        iFailureCount++;
                        resolve();
                    }
                }.bind(this));
                
                aPromises.push(oPromise);
            }.bind(this));
            
            return Promise.all(aPromises).then(function() {
                var sMessage = this._getResourceBundle().getText("msg.validationComplete", [iSuccessCount, iFailureCount]);
                MessageToast.show(sMessage);
            }.bind(this));
        },

        /**
         * @function _exportCatalog
         * @description Export catalog in specified format with security validation.
         * @param {string} sFormat - Export format (json, csv, xml)
         * @returns {Promise} Promise that resolves when export is complete
         * @private
         */
        _exportCatalog: function (sFormat) {
            return this._withErrorRecovery(function() {
                var oModel = this.base.getView().getModel();
                var exportData = { format: sFormat, includePrivate: false };
                var validation = SecurityUtils.validateExportData(exportData, false);
                
                if (!validation.valid) {
                    throw new Error(validation.error);
                }
                
                return SecurityUtils.secureCallFunction(oModel, "/exportCatalog", {
                    urlParameters: validation.sanitized,
                    success: function(oData) {
                        var sUrl = oData.downloadUrl;
                        if (sUrl) {
                            window.open(sUrl, "_blank");
                        }
                        MessageToast.show(this._getResourceBundle().getText("msg.exportStarted"));
                    }.bind(this),
                    error: function() {
                        throw new Error("Export operation failed");
                    }
                });
            }.bind(this), "ExportCatalog").catch(function(oError) {
                MessageToast.show(this._getResourceBundle().getText("error.exportFailed"));
            }.bind(this));
        },

        /**
         * @function formatStatus
         * @description Format status for display with internationalization.
         * @param {string} sStatus - Status value
         * @returns {string} Formatted status text
         */
        formatStatus: function (sStatus) {
            if (!sStatus) return "";
            return this._getResourceBundle().getText("status." + sStatus.toLowerCase(), sStatus);
        },

        /**
         * @function formatQualityState
         * @description Format quality score with appropriate semantic state.
         * @param {number} iScore - Quality score (0-100)
         * @returns {string} Semantic state (Success, Warning, Error)
         */
        formatQualityState: function (iScore) {
            if (!iScore || isNaN(iScore)) return "None";
            if (iScore >= 80) return "Success";
            if (iScore >= 60) return "Warning";
            return "Error";
        },
        
        /**
         * @function _getResourceBundle
         * @description Gets the resource bundle for internationalization.
         * @returns {sap.base.i18n.ResourceBundle} Resource bundle
         * @private
         */
        _getResourceBundle: function() {
            return this.base.getView().getModel("i18n").getResourceBundle();
        }
    });
});