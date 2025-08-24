sap.ui.define([
    "sap/ui/core/mvc/ControllerExtension",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Fragment",
    "sap/ui/model/json/JSONModel",
    "a2a/network/agent0/ext/utils/SecurityUtils"
], (ControllerExtension, MessageToast, MessageBox, Fragment, JSONModel, SecurityUtils) => {
    "use strict";

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
             * @description Initializes the controller extension with security utilities, device model,
             * dialog caching, and real-time updates.
             * @override
             */
            onInit() {
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
            onExit() {
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
        _initializeDeviceModel() {
            const oDeviceModel = new sap.ui.model.json.JSONModel(sap.ui.Device);
            this.base.getView().setModel(oDeviceModel, "device");
        },

        /**
         * @function _initializeDialogCache
         * @description Initializes dialog cache for performance.
         * @private
         */
        _initializeDialogCache() {
            this._dialogCache = {};
        },

        /**
         * @function _initializePerformanceOptimizations
         * @description Sets up performance optimization features.
         * @private
         */
        _initializePerformanceOptimizations() {
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
        _getOrCreateDialog(sDialogId, sFragmentName) {
            const that = this;
            if (this._dialogCache[sDialogId]) {
                return Promise.resolve(this._dialogCache[sDialogId]);
            }
            return Fragment.load({
                id: this.base.getView().getId(),
                name: sFragmentName,
                controller: this
            }).then((oDialog) => {
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
        _withErrorRecovery(fnOperation, sOperationName) {
            const that = this;
            const oConfig = this._errorRecoveryConfig;

            function attempt(nRetry) {
                return fnOperation().catch((oError) => {
                    if (nRetry < oConfig.maxRetries) {
                        const nDelay = oConfig.exponentialBackoff ?
                            oConfig.retryDelay * Math.pow(2, nRetry) : oConfig.retryDelay;

                        MessageToast.show(that._getResourceBundle().getText("recovery.retrying"));

                        return new Promise((resolve) => {
                            setTimeout(() => {
                                resolve(attempt(nRetry + 1));
                            }, nDelay);
                        });
                    }
                    MessageToast.show(that._getResourceBundle().getText("recovery.maxRetriesReached"));
                    throw oError;

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
        _throttle(fn, limit) {
            let inThrottle;
            return function() {
                const args = arguments;
                const context = this;
                if (!inThrottle) {
                    fn.apply(context, args);
                    inThrottle = true;
                    setTimeout(() => {
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
        _debounce(fn, delay) {
            let timeoutId;
            return function() {
                const args = arguments;
                const context = this;
                clearTimeout(timeoutId);
                timeoutId = setTimeout(() => {
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
        _enableDialogAccessibility(oDialog) {
            // Add ARIA attributes
            oDialog.addStyleClass("a2aAccessibleDialog");

            // Enable keyboard navigation
            oDialog.attachAfterOpen(() => {
                const oFirstInput = oDialog.$().find("input, button, select, textarea").first();
                if (oFirstInput.length) {
                    oFirstInput.focus();
                }
            });

            // Add escape key handler
            oDialog.addEventDelegate({
                onkeydown(oEvent) {
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
        _optimizeDialogForDevice(oDialog) {
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
        _startRealtimeProductUpdates() {
            if (this._ws) {return;}

            try {
                this._ws = SecurityUtils.createSecureWebSocket("wss://localhost:8000/dataproduct/updates", {
                    onmessage: function(event) {
                        const oData = JSON.parse(event.data);
                        this._handleProductUpdate(oData);
                    }.bind(this),
                    onerror: function(error) {
                        // console.warn("Secure WebSocket error:", error);
                        this._initializePolling();
                    }.bind(this)
                });

                if (this._ws) {
                    this._ws.onclose = function() {
                        setTimeout(() => this._startRealtimeProductUpdates(), 5000);
                    }.bind(this);
                }

            } catch (error) {
                // console.warn("WebSocket connection failed, falling back to polling");
                this._initializePolling();
            }
        },

        /**
         * @function _initializePolling
         * @description Initializes polling fallback for real-time updates.
         * @private
         */
        _initializePolling() {
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
        _handleProductUpdate(oData) {
            switch (oData.type) {
            case "PRODUCT_CREATED":
                MessageToast.show(this._getResourceBundle().getText("msg.productCreated"));
                this._refreshProductData();
                break;
            case "METADATA_UPDATED":
                MessageToast.show(this._getResourceBundle().getText("msg.metadataUpdated"));
                this._refreshProductData();
                break;
            case "VALIDATION_COMPLETED":
                this._updateValidationStatus(oData);
                break;
            case "QUALITY_ASSESSED":
                this._updateQualityMetrics(oData);
                break;
            }
        },

        /**
         * @function _refreshProductData
         * @description Refreshes the product data in the table.
         * @private
         */
        _refreshProductData() {
            const oTable = this.base.getView().byId("fe::table::DataProducts::LineItem");
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
        _updateValidationStatus(oData) {
            // Implementation for real-time validation status updates
        },

        /**
         * @function _updateQualityMetrics
         * @description Updates quality metrics in real-time.
         * @param {object} oData - Quality metrics data
         * @private
         */
        _updateQualityMetrics(oData) {
            // Implementation for real-time quality metrics updates
        },

        /**
         * @function _cleanupResources
         * @description Cleans up WebSocket connections and intervals.
         * @private
         */
        _cleanupResources() {
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
        _performSearch(sQuery) {
            // Implementation for search functionality
        },

        /**
         * @function _initializeModels
         * @description Initialize models for the view with comprehensive data structure.
         * @private
         */
        _initializeModels() {
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
        onCreateDataProduct() {
            this._withErrorRecovery(() => {
                return this._getOrCreateDialog("createDataProduct", "a2a.network.agent0.ext.fragment.CreateDataProduct")
                    .then((oDialog) => {
                        oDialog.open();
                    });
            }, "CreateDataProduct").catch((oError) => {
                MessageToast.show(this._getResourceBundle().getText("error.dialogLoadFailed"));
            });
        },

        /**
         * @function onImportMetadata
         * @description Handler for Import Metadata action with error recovery and caching.
         */
        onImportMetadata() {
            this._withErrorRecovery(() => {
                return this._getOrCreateDialog("importMetadata", "a2a.network.agent0.ext.fragment.ImportMetadata")
                    .then((oDialog) => {
                        oDialog.open();
                    });
            }, "ImportMetadata").catch((oError) => {
                MessageToast.show(this._getResourceBundle().getText("error.dialogLoadFailed"));
            });
        },

        /**
         * @function onOpenDashboard
         * @description Handler for Dashboard action with throttled data loading.
         */
        onOpenDashboard() {
            this._withErrorRecovery(() => {
                return this._getOrCreateDialog("dashboard", "a2a.network.agent0.ext.fragment.DataProductDashboard")
                    .then((oDialog) => {
                        this._throttledDashboardUpdate();
                        oDialog.open();
                    });
            }, "OpenDashboard").catch((oError) => {
                MessageToast.show(this._getResourceBundle().getText("error.dashboardLoadFailed"));
            });
        },

        /**
         * @function onValidateMetadata
         * @description Handler for Validate Metadata action with comprehensive error handling.
         * @param {sap.ui.base.Event} oEvent - Event object
         */
        onValidateMetadata(oEvent) {
            const oTable = this.base.getView().byId("fe::table::DataProducts::LineItem");
            const aSelectedContexts = oTable.getSelectedContexts();

            if (aSelectedContexts.length === 0) {
                MessageToast.show(this._getResourceBundle().getText("msg.selectDataProduct"));
                return;
            }

            this._withErrorRecovery(() => {
                return this._validateSelectedProducts(aSelectedContexts);
            }, "ValidateMetadata").catch((oError) => {
                MessageToast.show(this._getResourceBundle().getText("error.validationFailed"));
            });
        },

        /**
         * @function onBulkUpdate
         * @description Handler for Bulk Update action with selection validation and dialog caching.
         */
        onBulkUpdate() {
            const oTable = this.base.getView().byId("fe::table::DataProducts::LineItem");
            const aSelectedContexts = oTable.getSelectedContexts();

            if (aSelectedContexts.length === 0) {
                MessageToast.show(this._getResourceBundle().getText("msg.selectProductsToUpdate"));
                return;
            }

            this._withErrorRecovery(() => {
                return this._getOrCreateDialog("bulkUpdate", "a2a.network.agent0.ext.fragment.BulkUpdate")
                    .then((oDialog) => {
                        oDialog.open();
                    });
            }, "BulkUpdate").catch((oError) => {
                MessageToast.show(this._getResourceBundle().getText("error.bulkUpdateFailed"));
            });
        },

        /**
         * @function onExportCatalog
         * @description Handler for Export Catalog action with format selection dialog.
         */
        onExportCatalog() {
            this._withErrorRecovery(() => {
                const oResourceBundle = this._getResourceBundle();
                const oDialog = new sap.m.Dialog({
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
                        press: function() {
                            const iSelectedIndex = oDialog.getContent()[1].getSelectedIndex();
                            const aFormats = ["json", "csv", "xml"];
                            this._exportCatalog(aFormats[iSelectedIndex]);
                            oDialog.close();
                        }.bind(this)
                    }),
                    endButton: new sap.m.Button({
                        text: oResourceBundle.getText("btn.cancel"),
                        press() {
                            oDialog.close();
                        }
                    }),
                    afterClose() {
                        oDialog.destroy();
                    }
                });

                this._enableDialogAccessibility(oDialog);
                this._optimizeDialogForDevice(oDialog);
                oDialog.open();
                return Promise.resolve();
            }, "ExportCatalog").catch((oError) => {
                MessageToast.show(this._getResourceBundle().getText("error.exportDialogFailed"));
            });
        },

        /**
         * @function _loadDashboardData
         * @description Load dashboard data with error recovery and security validation.
         * @returns {Promise} Promise that resolves when dashboard data is loaded
         * @private
         */
        _loadDashboardData() {
            const oViewModel = this.base.getView().getModel("viewModel");
            oViewModel.setProperty("/busy", true);

            return this._withErrorRecovery(() => {
                const oModel = this.base.getView().getModel();
                return SecurityUtils.secureCallFunction(oModel, "/GetDashboardMetrics", {
                    success: function(oData) {
                        oViewModel.setProperty("/dashboardData", oData);
                    }.bind(this),
                    error(oError) {
                        throw new Error("Dashboard data loading failed");
                    }
                });
            }, "LoadDashboardData").catch((oError) => {
                MessageToast.show(this._getResourceBundle().getText("error.loadingDashboardData"));
            }).finally(() => {
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
        _validateSelectedProducts(aContexts) {
            const oModel = this.base.getView().getModel();
            let iSuccessCount = 0;
            let iFailureCount = 0;
            const aPromises = [];

            aContexts.forEach((oContext) => {
                const oPromise = new Promise((resolve, reject) => {
                    try {
                        const sPath = oContext.getPath();
                        const oData = oContext.getObject();
                        const validation = SecurityUtils.validateDublinCore(oData.dublinCore);

                        if (!validation.valid) {
                            throw new Error(validation.error);
                        }

                        SecurityUtils.secureCallFunction(oModel, `${sPath }/validateMetadata`, {
                            success() {
                                iSuccessCount++;
                                resolve();
                            },
                            error() {
                                iFailureCount++;
                                resolve(); // Don't reject, just count failures
                            }
                        });
                    } catch (error) {
                        iFailureCount++;
                        resolve();
                    }
                });

                aPromises.push(oPromise);
            });

            return Promise.all(aPromises).then(() => {
                const sMessage = this._getResourceBundle().getText("msg.validationComplete", [iSuccessCount, iFailureCount]);
                MessageToast.show(sMessage);
            });
        },

        /**
         * @function _exportCatalog
         * @description Export catalog in specified format with security validation.
         * @param {string} sFormat - Export format (json, csv, xml)
         * @returns {Promise} Promise that resolves when export is complete
         * @private
         */
        _exportCatalog(sFormat) {
            return this._withErrorRecovery(() => {
                const oModel = this.base.getView().getModel();
                const exportData = { format: sFormat, includePrivate: false };
                const validation = SecurityUtils.validateExportData(exportData, false);

                if (!validation.valid) {
                    throw new Error(validation.error);
                }

                return SecurityUtils.secureCallFunction(oModel, "/exportCatalog", {
                    urlParameters: validation.sanitized,
                    success: function(oData) {
                        const sUrl = oData.downloadUrl;
                        if (sUrl) {
                            window.open(sUrl, "_blank");
                        }
                        MessageToast.show(this._getResourceBundle().getText("msg.exportStarted"));
                    }.bind(this),
                    error() {
                        throw new Error("Export operation failed");
                    }
                });
            }, "ExportCatalog").catch((oError) => {
                MessageToast.show(this._getResourceBundle().getText("error.exportFailed"));
            });
        },

        /**
         * @function formatStatus
         * @description Format status for display with internationalization.
         * @param {string} sStatus - Status value
         * @returns {string} Formatted status text
         */
        formatStatus(sStatus) {
            if (!sStatus) {return "";}
            return this._getResourceBundle().getText(`status.${ sStatus.toLowerCase()}`, sStatus);
        },

        /**
         * @function formatQualityState
         * @description Format quality score with appropriate semantic state.
         * @param {number} iScore - Quality score (0-100)
         * @returns {string} Semantic state (Success, Warning, Error)
         */
        formatQualityState(iScore) {
            if (!iScore || isNaN(iScore)) {return "None";}
            if (iScore >= 80) {return "Success";}
            if (iScore >= 60) {return "Warning";}
            return "Error";
        },

        /**
         * @function _getResourceBundle
         * @description Gets the resource bundle for internationalization.
         * @returns {sap.base.i18n.ResourceBundle} Resource bundle
         * @private
         */
        _getResourceBundle() {
            return this.base.getView().getModel("i18n").getResourceBundle();
        }
    });
});