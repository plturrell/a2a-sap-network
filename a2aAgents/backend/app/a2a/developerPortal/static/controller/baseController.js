sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/core/routing/History",
    "sap/ui/core/UIComponent",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/model/json/JSONModel"
], (Controller, History, UIComponent, MessageToast, MessageBox, JSONModel) => {
    "use strict";

    /**
     * Enhanced Base Controller with Offline Support
     * Provides common functionality for all controllers in the A2A Portal
     */
    return Controller.extend("sap.a2a.controller.BaseController", {

        /**
         * Convenience method for accessing the router in every controller of the application.
         * @public
         * @returns {sap.ui.core.routing.Router} the router for this component
         */
        getRouter: function () {
            return this.getOwnerComponent().getRouter();
        },

        /**
         * Convenience method for getting the view model by name in every controller of the application.
         * @public
         * @param {string} [sName] the model name
         * @returns {sap.ui.model.Model} the model instance
         */
        getModel: function (sName) {
            return this.getView().getModel(sName);
        },

        /**
         * Convenience method for setting the view model in every controller of the application.
         * @public
         * @param {sap.ui.model.Model} oModel the model instance
         * @param {string} [sName] the model name
         * @returns {sap.ui.core.mvc.View} the view instance
         */
        setModel: function (oModel, sName) {
            return this.getView().setModel(oModel, sName);
        },

        /**
         * Convenience method for getting the resource bundle.
         * @public
         * @returns {sap.ui.model.resource.ResourceModel} the resourceModel of the component
         */
        getResourceBundle: function () {
            return this.getOwnerComponent().getModel("i18n").getResourceBundle();
        },

        /**
         * Method for navigation to specific view
         * @public
         * @param {string} psTarget Parameter containing the string for the target navigation
         * @param {mapping} pmParameters? Parameters for navigation
         * @param {boolean} pbReplace? Defines if the hash should be replaced (no browser history entry) or set (browser history entry)
         */
        navTo: function (psTarget, pmParameters, pbReplace) {
            this.getRouter().navTo(psTarget, pmParameters, pbReplace);
        },

        /**
         * Navigate back to previous page
         * @public
         */
        onNavBack: function () {
            const sPreviousHash = History.getInstance().getPreviousHash();
            
            if (sPreviousHash !== undefined) {
                window.history.go(-1);
            } else {
                this.getRouter().navTo("home", {}, true /*no history*/);
            }
        },

        /**
         * Show busy indicator
         * @public
         * @param {number} iDelay delay in ms
         */
        showBusyIndicator: function (iDelay) {
            sap.ui.core.BusyIndicator.show(iDelay || 0);
        },

        /**
         * Hide busy indicator
         * @public
         */
        hideBusyIndicator: function () {
            sap.ui.core.BusyIndicator.hide();
        },

        /* =========================================================== */
        /* Enhanced Offline Support Methods                           */
        /* =========================================================== */

        /**
         * Initialize offline capabilities for this controller
         * @protected
         */
        initOfflineSupport: function () {
            if (!this._offlineManager) {
                this._offlineManager = sap.a2a.utils.OfflineManager;
            }
            
            // Setup offline event listeners
            this._setupOfflineEventListeners();
            
            // Initialize offline UI state
            this._initializeOfflineUI();
        },

        /**
         * Setup offline event listeners
         * @private
         */
        _setupOfflineEventListeners: function () {
            // Network status changes
            document.addEventListener('a2a.offline.networkOnline', (event) => {
                this._onNetworkOnline(event.detail);
            });
            
            document.addEventListener('a2a.offline.networkOffline', (event) => {
                this._onNetworkOffline(event.detail);
            });
            
            // Sync events
            document.addEventListener('a2a.offline.syncComplete', (event) => {
                this._onSyncComplete(event.detail);
            });
            
            document.addEventListener('a2a.offline.syncError', (event) => {
                this._onSyncError(event.detail);
            });
            
            // Data events
            document.addEventListener('a2a.offline.offlineDataLoaded', (event) => {
                this._onOfflineDataLoaded(event.detail);
            });
        },

        /**
         * Initialize offline UI state
         * @private
         */
        _initializeOfflineUI: function () {
            // Create offline status model if not exists
            if (!this.getModel("offline")) {
                const oOfflineModel = new JSONModel({
                    isOnline: this._offlineManager.isOnline(),
                    syncStatus: this._offlineManager.getSyncStatus(),
                    lastSync: null,
                    pendingChanges: 0,
                    showOfflineMessage: false
                });
                this.setModel(oOfflineModel, "offline");
            }
            
            // Update initial state
            this._updateOfflineStatus();
        },

        /**
         * Handle network online event
         * @protected
         * @param {object} oEventData event data
         */
        _onNetworkOnline: function (_oEventData) {
            this._updateOfflineStatus();
            
            // Show connection restored message
            MessageToast.show(this.getResourceBundle().getText("offline.connectionRestored"), {
                duration: 3000
            });
            
            // Refresh data if needed
            this._refreshDataAfterOnline();
        },

        /**
         * Handle network offline event
         * @protected
         * @param {object} oEventData event data
         */
        _onNetworkOffline: function (_oEventData) {
            this._updateOfflineStatus();
            
            // Show offline message
            MessageToast.show(this.getResourceBundle().getText("offline.workingOffline"), {
                duration: 5000
            });
        },

        /**
         * Handle sync completion
         * @protected
         * @param {object} oEventData event data
         */
        _onSyncComplete: function (_oEventData) {
            this._updateOfflineStatus();
            
            // Show success message
            MessageToast.show(this.getResourceBundle().getText("offline.syncComplete"));
            
            // Refresh UI data
            this._refreshAfterSync();
        },

        /**
         * Handle sync error
         * @protected
         * @param {object} oEventData event data
         */
        _onSyncError: function (oEventData) {
            this._updateOfflineStatus();
            
            // Show error message
            MessageToast.show(this.getResourceBundle().getText("offline.syncError", [oEventData.error]), {
                duration: 5000
            });
        },

        /**
         * Handle offline data loaded
         * @protected
         * @param {object} oEventData event data
         */
        _onOfflineDataLoaded: function (oEventData) {
            // Update counts in offline model
            const oOfflineModel = this.getModel("offline");
            if (oOfflineModel) {
                oOfflineModel.setProperty("/offlineData", {
                    projectsCount: oEventData.projectsCount,
                    agentsCount: oEventData.agentsCount,
                    workflowsCount: oEventData.workflowsCount
                });
            }
        },

        /**
         * Update offline status in UI
         * @private
         */
        _updateOfflineStatus: function () {
            const oOfflineModel = this.getModel("offline");
            if (oOfflineModel && this._offlineManager) {
                const oSyncStatus = this._offlineManager.getSyncStatus();
                
                oOfflineModel.setData({
                    isOnline: this._offlineManager.isOnline(),
                    syncStatus: oSyncStatus,
                    lastSync: oSyncStatus.lastSyncTime,
                    pendingChanges: oSyncStatus.queueLength,
                    showOfflineMessage: !this._offlineManager.isOnline(),
                    conflicts: oSyncStatus.conflictsCount
                });
            }
        },

        /**
         * Refresh data after coming online
         * @protected
         */
        _refreshDataAfterOnline: function () {
            // Override in specific controllers
            // eslint-disable-next-line no-console
            // eslint-disable-next-line no-console
            console.log("BaseController: Refreshing data after coming online");
        },

        /**
         * Refresh UI after synchronization
         * @protected
         */
        _refreshAfterSync: function () {
            // Override in specific controllers
            // eslint-disable-next-line no-console
            // eslint-disable-next-line no-console
            console.log("BaseController: Refreshing UI after sync");
        },

        /* =========================================================== */
        /* Offline Data Operations                                     */
        /* =========================================================== */

        /**
         * Load data with offline support
         * @protected
         * @param {string} sEntityType entity type (projects, agents, etc.)
         * @param {object} oOptions loading options
         * @returns {Promise} promise resolving to data
         */
        loadDataWithOfflineSupport: function (sEntityType, oOptions = {}) {
            return new Promise((resolve, reject) => {
                if (this._offlineManager.isOnline()) {
                    // Online: fetch from server
                    this._loadDataFromServer(sEntityType, oOptions)
                        .then((oData) => {
                            // Store for offline use
                            this._offlineManager.storeOfflineData(sEntityType, oData.results || oData);
                            resolve(oData);
                        })
                        .catch((oError) => {
                            // Fallback to offline data
                            console.warn("Server request failed, using offline data:", oError);
                            this._loadDataFromOffline(sEntityType, oOptions)
                                .then(resolve)
                                .catch(reject);
                        });
                } else {
                    // Offline: use cached data
                    this._loadDataFromOffline(sEntityType, oOptions)
                        .then(resolve)
                        .catch(reject);
                }
            });
        },

        /**
         * Load data from server
         * @private
         * @param {string} sEntityType entity type
         * @param {object} oOptions options
         * @returns {Promise} promise
         */
        _loadDataFromServer: function (sEntityType, oOptions) {
            return new Promise((resolve, reject) => {
                const oModel = this.getModel();
                const sPath = `/${sEntityType}`;
                
                oModel.read(sPath, {
                    urlParameters: oOptions.urlParameters || {},
                    filters: oOptions.filters || [],
                    sorters: oOptions.sorters || [],
                    success: resolve,
                    error: reject
                });
            });
        },

        /**
         * Load data from offline storage
         * @private
         * @param {string} sEntityType entity type
         * @param {object} oOptions options
         * @returns {Promise} promise
         */
        _loadDataFromOffline: function (sEntityType, oOptions) {
            return Promise.resolve({
                results: this._offlineManager.getOfflineData(sEntityType, oOptions)
            });
        },

        /**
         * Create entity with offline support
         * @protected
         * @param {string} sEntityType entity type
         * @param {object} oData entity data
         * @returns {Promise} promise
         */
        createEntityWithOfflineSupport: function (sEntityType, oData) {
            if (this._offlineManager.isOnline()) {
                return this._createEntityOnServer(sEntityType, oData);
            } else {
                return this._createEntityOffline(sEntityType, oData);
            }
        },

        /**
         * Create entity on server
         * @private
         * @param {string} sEntityType entity type
         * @param {object} oData entity data
         * @returns {Promise} promise
         */
        _createEntityOnServer: function (sEntityType, oData) {
            return new Promise((resolve, reject) => {
                const oModel = this.getModel();
                const sPath = `/${sEntityType}`;
                
                oModel.create(sPath, oData, {
                    success: (oCreatedData) => {
                        // Update offline storage
                        this._offlineManager.storeOfflineData(sEntityType, oCreatedData);
                        resolve(oCreatedData);
                    },
                    error: reject
                });
            });
        },

        /**
         * Create entity offline (queue for sync)
         * @private
         * @param {string} sEntityType entity type
         * @param {object} oData entity data
         * @returns {Promise} promise
         */
        _createEntityOffline: function (sEntityType, oData) {
            // Generate temporary ID
            const sTempId = this._generateTempId();
            const oEntityData = { ...oData, ID: sTempId, _tempId: true };
            
            // Queue for synchronization
            const sQueueId = this._offlineManager.queueOperation({
                type: 'CREATE',
                entityType: sEntityType,
                entityId: sTempId,
                data: oEntityData
            });
            
            // Store locally
            return this._offlineManager.storeOfflineData(sEntityType, oEntityData)
                .then(() => {
                    return { ...oEntityData, _queueId: sQueueId };
                });
        },

        /**
         * Update entity with offline support
         * @protected
         * @param {string} sEntityType entity type
         * @param {string} sEntityId entity ID
         * @param {object} oData updated data
         * @returns {Promise} promise
         */
        updateEntityWithOfflineSupport: function (sEntityType, sEntityId, oData) {
            if (this._offlineManager.isOnline()) {
                return this._updateEntityOnServer(sEntityType, sEntityId, oData);
            } else {
                return this._updateEntityOffline(sEntityType, sEntityId, oData);
            }
        },

        /**
         * Update entity on server
         * @private
         */
        _updateEntityOnServer: function (sEntityType, sEntityId, oData) {
            return new Promise((resolve, reject) => {
                const oModel = this.getModel();
                const sPath = `/${sEntityType}('${sEntityId}')`;
                
                oModel.update(sPath, oData, {
                    success: (oUpdatedData) => {
                        // Update offline storage
                        this._offlineManager.storeOfflineData(sEntityType, oUpdatedData);
                        resolve(oUpdatedData);
                    },
                    error: reject
                });
            });
        },

        /**
         * Update entity offline (queue for sync)
         * @private
         */
        _updateEntityOffline: function (sEntityType, sEntityId, oData) {
            // Queue for synchronization
            const sQueueId = this._offlineManager.queueOperation({
                type: 'UPDATE',
                entityType: sEntityType,
                entityId: sEntityId,
                data: oData
            });
            
            // Update locally
            const oUpdatedData = { ...oData, ID: sEntityId, _modified: true };
            
            return this._offlineManager.storeOfflineData(sEntityType, oUpdatedData)
                .then(() => {
                    return { ...oUpdatedData, _queueId: sQueueId };
                });
        },

        /**
         * Delete entity with offline support
         * @protected
         * @param {string} sEntityType entity type
         * @param {string} sEntityId entity ID
         * @returns {Promise} promise
         */
        deleteEntityWithOfflineSupport: function (sEntityType, sEntityId) {
            if (this._offlineManager.isOnline()) {
                return this._deleteEntityOnServer(sEntityType, sEntityId);
            } else {
                return this._deleteEntityOffline(sEntityType, sEntityId);
            }
        },

        /**
         * Delete entity on server
         * @private
         */
        _deleteEntityOnServer: function (sEntityType, sEntityId) {
            return new Promise((resolve, reject) => {
                const oModel = this.getModel();
                const sPath = `/${sEntityType}('${sEntityId}')`;
                
                oModel.remove(sPath, {
                    success: resolve,
                    error: reject
                });
            });
        },

        /**
         * Delete entity offline (queue for sync)
         * @private
         */
        _deleteEntityOffline: function (sEntityType, sEntityId) {
            // Queue for synchronization
            const sQueueId = this._offlineManager.queueOperation({
                type: 'DELETE',
                entityType: sEntityType,
                entityId: sEntityId
            });
            
            return Promise.resolve({ success: true, _queueId: sQueueId });
        },

        /* =========================================================== */
        /* Utility Methods                                            */
        /* =========================================================== */

        /**
         * Generate temporary ID for offline entities
         * @private
         * @returns {string} temporary ID
         */
        _generateTempId: function () {
            return `temp_${  Date.now().toString(36)  }${Math.random().toString(36).substr(2, 9)}`;
        },

        /**
         * Check if entity is temporary (created offline)
         * @protected
         * @param {object} oEntity entity object
         * @returns {boolean} true if temporary
         */
        isTemporaryEntity: function (oEntity) {
            return oEntity && (oEntity._tempId === true || (oEntity.ID && oEntity.ID.startsWith('temp_')));
        },

        /**
         * Check if entity is modified offline
         * @protected
         * @param {object} oEntity entity object
         * @returns {boolean} true if modified offline
         */
        isModifiedOffline: function (oEntity) {
            return oEntity && oEntity._modified === true;
        },

        /**
         * Show offline indicator in UI
         * @protected
         * @param {boolean} bShow show/hide indicator
         */
        showOfflineIndicator: function (bShow) {
            const oOfflineModel = this.getModel("offline");
            if (oOfflineModel) {
                oOfflineModel.setProperty("/showOfflineMessage", bShow);
            }
        },

        /**
         * Force synchronization
         * @public
         */
        forceSynchronization: function () {
            if (this._offlineManager) {
                this.showBusyIndicator();
                
                this._offlineManager.synchronize()
                    .then(() => {
                        MessageToast.show(this.getResourceBundle().getText("offline.syncForceComplete"));
                        this._refreshAfterSync();
                    })
                    .catch((oError) => {
                        MessageBox.error(this.getResourceBundle().getText("offline.syncForceError", [oError.message]));
                    })
                    .finally(() => {
                        this.hideBusyIndicator();
                    });
            }
        },

        /**
         * Clear offline data
         * @public
         * @param {string} sEntityType entity type (optional, clears all if not specified)
         */
        clearOfflineData: function (sEntityType) {
            const sMessage = sEntityType ? 
                this.getResourceBundle().getText("offline.clearEntityConfirm", [sEntityType]) :
                this.getResourceBundle().getText("offline.clearAllConfirm");
            
            MessageBox.confirm(sMessage, {
                title: this.getResourceBundle().getText("offline.clearDataTitle"),
                onClose: (sAction) => {
                    if (sAction === MessageBox.Action.OK && this._offlineManager) {
                        this._offlineManager.clearOfflineData(sEntityType)
                            .then(() => {
                                MessageToast.show(this.getResourceBundle().getText("offline.dataCleared"));
                                this._updateOfflineStatus();
                            })
                            .catch((oError) => {
                                MessageBox.error(this.getResourceBundle().getText("offline.clearError", [oError.message]));
                            });
                    }
                },
                emphasizedAction: MessageBox.Action.OK
            });
        },

        /* =========================================================== */
        /* Offline Status Bar Event Handlers                         */
        /* =========================================================== */

        /**
         * Handle sync now button press
         */
        onSyncNow: function () {
            this.forceSynchronization();
        },

        /**
         * Handle view sync queue menu item
         */
        onViewSyncQueue: function () {
            if (this._offlineManager) {
                this._offlineManager.getSyncQueue()
                    .then((aSyncQueue) => {
                        this._openSyncQueueDialog(aSyncQueue);
                    })
                    .catch((oError) => {
                        MessageBox.error(this.getResourceBundle().getText("offline.viewQueueError", [oError.message]));
                    });
            }
        },

        /**
         * Handle resolve conflicts menu item
         */
        onResolveConflicts: function () {
            if (this._offlineManager) {
                this._offlineManager.getConflicts()
                    .then((aConflicts) => {
                        if (aConflicts.length === 0) {
                            MessageToast.show(this.getResourceBundle().getText("offline.noConflictsFound"));
                            return;
                        }
                        this._openConflictResolutionDialog(aConflicts);
                    })
                    .catch((oError) => {
                        MessageBox.error(this.getResourceBundle().getText("offline.conflictsError", [oError.message]));
                    });
            }
        },

        /**
         * Handle clear offline data menu item
         */
        onClearOfflineData: function () {
            this.clearOfflineData();
        },

        /**
         * Handle offline settings menu item
         */
        onOfflineSettings: function () {
            this._openOfflineSettingsDialog();
        },

        /**
         * Handle close offline bar button
         */
        onCloseOfflineBar: function () {
            this.showOfflineIndicator(false);
        },

        /**
         * Format offline message
         */
        formatOfflineMessage: function (iPendingChanges, _sMessageWithChanges, _sMessageWithoutChanges) {
            if (iPendingChanges > 0) {
                return this.getResourceBundle().getText("offline.workingOfflineWithChanges", [iPendingChanges]);
            }
            return this.getResourceBundle().getText("offline.workingOffline");
        },

        /**
         * Format pending changes text
         */
        formatPendingChanges: function (iCount, _sTemplate) {
            return this.getResourceBundle().getText("offline.pendingChangesCount", [iCount]);
        },

        /**
         * Format conflicts text
         */
        formatConflicts: function (iCount, _sTemplate) {
            return this.getResourceBundle().getText("offline.conflictsCount", [iCount]);
        },

        /**
         * Format last sync time
         */
        formatLastSync: function (oLastSync, _sTemplate) {
            if (!oLastSync) {
return this.getResourceBundle().getText("offline.neverSynced");
}
            
            const oDateFormat = sap.ui.core.format.DateFormat.getDateTimeInstance({
                style: "medium"
            });
            return this.getResourceBundle().getText("offline.lastSyncTime", [oDateFormat.format(new Date(oLastSync))]);
        },

        /* =========================================================== */
        /* Private Dialog Methods                                     */
        /* =========================================================== */

        /**
         * Open conflict resolution dialog
         * @private
         */
        _openConflictResolutionDialog: function (aConflicts) {
            sap.ui.require([
                "sap/a2a/controller/ConflictResolutionController"
            ], (ConflictResolutionController) => {
                if (!this._oConflictController) {
                    this._oConflictController = new ConflictResolutionController();
                    this._oConflictController.connectToView(this.getView());
                }
                this._oConflictController.openConflictResolutionDialog(aConflicts);
            });
        },

        /**
         * Open sync queue dialog
         * @private
         */
        _openSyncQueueDialog: function (aSyncQueue) {
            const _oSyncQueueModel = new sap.ui.model.json.JSONModel({
                syncQueue: aSyncQueue
            });
            
            // Simple message box for now - could be enhanced with a proper dialog
            const aSyncMessages = aSyncQueue.map(oItem => {
                return `${oItem.operation} ${oItem.entityType}: ${oItem.status}`;
            });
            
            MessageBox.information(
                aSyncMessages.join("\n") || this.getResourceBundle().getText("offline.syncQueueEmpty"),
                {
                    title: this.getResourceBundle().getText("offline.syncQueueTitle")
                }
            );
        },

        /**
         * Open offline settings dialog
         * @private
         */
        _openOfflineSettingsDialog: function () {
            // Placeholder for offline settings dialog
            MessageToast.show(this.getResourceBundle().getText("offline.settingsNotImplemented"));
        }
    });
});