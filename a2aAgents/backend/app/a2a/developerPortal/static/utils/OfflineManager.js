sap.ui.define([
    "sap/ui/base/Object",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast",
    "sap/m/MessageBox"
], (BaseObject, JSONModel, MessageToast, MessageBox) => {
    "use strict";
/* global indexedDB, CustomEvent */

    /**
     * Enterprise Offline Manager for SAP A2A Developer Portal
     * Provides comprehensive offline capabilities with data synchronization
     */
    const OfflineManager = BaseObject.extend("sap.a2a.utils.OfflineManager", {

        constructor: function () {
            // eslint-disable-next-line prefer-rest-params
            BaseObject.prototype.constructor.apply(this, arguments);
            
            this._isOnline = navigator.onLine;
            this._syncQueue = [];
            this._offlineData = {};
            this._conflictQueue = [];
            this._lastSyncTime = null;
            this._syncInProgress = false;
            
            // Configuration
            this._config = {
                storageQuota: 50 * 1024 * 1024, // 50MB
                syncInterval: 30000, // 30 seconds
                retryInterval: 5000, // 5 seconds
                maxRetries: 5,
                criticalDataTypes: ['projects', 'agents', 'workflows'],
                offlineTTL: 24 * 60 * 60 * 1000, // 24 hours
                conflictResolution: 'client-wins' // or 'server-wins', 'manual'
            };
            
            this._initializeOfflineCapabilities();
        },

        /**
         * Initialize offline capabilities
         */
        _initializeOfflineCapabilities: function () {
            // Initialize storage
            this._initializeStorage();
            
            // Setup network monitoring
            this._setupNetworkMonitoring();
            
            // Register service worker
            this._registerServiceWorker();
            
            // Setup periodic sync
            this._setupPeriodicSync();
            
            // Load offline data
            this._loadOfflineData();
            
            // Setup visibility change handling
            this._setupVisibilityChangeHandling();
            
             
            
            // eslint-disable-next-line no-console
            
             
            
            // eslint-disable-next-line no-console
            console.log("Offline Manager initialized successfully");
        },

        /**
         * Initialize storage systems
         */
        _initializeStorage: function () {
            // Initialize IndexedDB
            this._initializeIndexedDB();
            
            // Request persistent storage
            if (navigator.storage && navigator.storage.persist) {
                navigator.storage.persist().then(granted => {
                    if (granted) {
                        // eslint-disable-next-line no-console
                        // eslint-disable-next-line no-console
                        console.log("Persistent storage granted");
                    } else {
                        console.warn("Persistent storage not granted");
                    }
                });
            }
            
            // Check storage quota
            this._checkStorageQuota();
        },

        /**
         * Initialize IndexedDB for offline data storage
         */
        _initializeIndexedDB: function () {
            return new Promise((resolve, reject) => {
                const request = indexedDB.open("A2APortalOffline", 3);
                
                request.onerror = () => {
                    console.error("Failed to open IndexedDB:", request.error);
                    reject(request.error);
                };
                
                request.onsuccess = (event) => {
                    this._db = event.target.result;
                    // eslint-disable-next-line no-console
                    // eslint-disable-next-line no-console
                    console.log("IndexedDB initialized successfully");
                    resolve(this._db);
                };
                
                request.onupgradeneeded = (event) => {
                    const db = event.target.result;
                    
                    // Create object stores
                    this._createObjectStores(db);
                };
            });
        },

        /**
         * Create IndexedDB object stores
         */
        _createObjectStores: function (db) {
            const stores = [
                { name: 'projects', keyPath: 'ID', indexes: ['status', 'priority', 'modifiedAt'] },
                { name: 'agents', keyPath: 'ID', indexes: ['projectId', 'type', 'status'] },
                { name: 'workflows', keyPath: 'ID', indexes: ['projectId', 'status'] },
                { name: 'templates', keyPath: 'ID', indexes: ['category', 'isPublic'] },
                { name: 'syncQueue', keyPath: 'id', indexes: ['timestamp', 'status'] },
                { name: 'conflicts', keyPath: 'id', indexes: ['entityType', 'entityId'] },
                { name: 'metadata', keyPath: 'key' }
            ];
            
            stores.forEach(storeConfig => {
                if (db.objectStoreNames.contains(storeConfig.name)) {
                    // Delete existing store to recreate with new schema
                    db.deleteObjectStore(storeConfig.name);
                }
                
                const objectStore = db.createObjectStore(storeConfig.name, {
                    keyPath: storeConfig.keyPath,
                    autoIncrement: storeConfig.keyPath === 'id'
                });
                
                // Create indexes
                if (storeConfig.indexes) {
                    storeConfig.indexes.forEach(index => {
                        objectStore.createIndex(index, index);
                    });
                }
            });
        },

        /**
         * Setup network monitoring
         */
        _setupNetworkMonitoring: function () {
            // Listen for online/offline events
            window.addEventListener('online', () => {
                this._handleOnlineEvent();
            });
            
            window.addEventListener('offline', () => {
                this._handleOfflineEvent();
            });
            
            // Periodic connectivity check
            this._startConnectivityCheck();
        },

        /**
         * Handle online event
         */
        _handleOnlineEvent: function () {
            // eslint-disable-next-line no-console
            // eslint-disable-next-line no-console
            console.log("Network connectivity restored");
            this._isOnline = true;
            
            // Update UI
            this._updateNetworkStatus(true);
            
            // Start synchronization
            this._startSynchronization();
            
            // Show notification
            MessageToast.show("Connection restored. Synchronizing data...");
            
            // Fire online event
            this._fireEvent('networkOnline', {
                timestamp: new Date()
            });
        },

        /**
         * Handle offline event
         */
        _handleOfflineEvent: function () {
            // eslint-disable-next-line no-console
            // eslint-disable-next-line no-console
            console.log("Network connectivity lost");
            this._isOnline = false;
            
            // Update UI
            this._updateNetworkStatus(false);
            
            // Show notification
            MessageToast.show("Working offline. Changes will sync when connected.");
            
            // Fire offline event
            this._fireEvent('networkOffline', {
                timestamp: new Date()
            });
        },

        /**
         * Start periodic connectivity check
         */
        _startConnectivityCheck: function () {
            setInterval(() => {
                this._checkConnectivity();
            }, 10000); // Check every 10 seconds
        },

        /**
         * Check actual connectivity (not just navigator.onLine)
         */
        _checkConnectivity: function () {
            if (!this._isOnline) {
return;
}
            
            // Ping server to verify connectivity
            fetch('/health', {
                method: 'GET',
                headers: {
                    'Cache-Control': 'no-cache',
                    'X-Connectivity-Check': 'true'
                }
            })
            .then(response => {
                if (response.ok && this._isOnline !== true) {
                    this._handleOnlineEvent();
                }
            })
            .catch(_error => {
                if (this._isOnline !== false) {
                    this._handleOfflineEvent();
                }
            });
        },

        /**
         * Register service worker for offline caching
         */
        _registerServiceWorker: function () {
            if ('serviceWorker' in navigator) {
                navigator.serviceWorker.register('/sw.js', {
                    scope: '/'
                })
                .then(registration => {
                    // eslint-disable-next-line no-console
                    // eslint-disable-next-line no-console
                    console.log('Service Worker registered successfully:', registration.scope);
                    this._serviceWorkerRegistration = registration;
                    
                    // Handle service worker updates
                    registration.addEventListener('updatefound', () => {
                        const newWorker = registration.installing;
                        newWorker.addEventListener('statechange', () => {
                            if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
                                this._handleServiceWorkerUpdate();
                            }
                        });
                    });
                })
                .catch(error => {
                    console.error('Service Worker registration failed:', error);
                });
            }
        },

        /**
         * Handle service worker updates
         */
        _handleServiceWorkerUpdate: function () {
            MessageBox.confirm(
                "A new version of the application is available. Reload to update?",
                {
                    title: "Update Available",
                    onClose: (sAction) => {
                        if (sAction === MessageBox.Action.OK) {
                            window.location.reload();
                        }
                    },
                    emphasizedAction: MessageBox.Action.OK
                }
            );
        },

        /**
         * Setup periodic synchronization
         */
        _setupPeriodicSync: function () {
            // Background sync when online
            setInterval(() => {
                if (this._isOnline && !this._syncInProgress) {
                    this._synchronizeData();
                }
            }, this._config.syncInterval);
            
            // Register background sync if supported
            if ('serviceWorker' in navigator && 'sync' in window.ServiceWorkerRegistration.prototype) {
                this._registerBackgroundSync();
            }
        },

        /**
         * Register background sync
         */
        _registerBackgroundSync: function () {
            navigator.serviceWorker.ready.then(registration => {
                return registration.sync.register('background-sync');
            }).then(() => {
                // eslint-disable-next-line no-console
                // eslint-disable-next-line no-console
                console.log('Background sync registered');
            }).catch(error => {
                console.error('Background sync registration failed:', error);
            });
        },

        /**
         * Setup visibility change handling
         */
        _setupVisibilityChangeHandling: function () {
            document.addEventListener('visibilitychange', () => {
                if (!document.hidden && this._isOnline) {
                    // App became visible, sync data
                    this._synchronizeData();
                }
            });
        },

        /**
         * Load offline data from storage
         */
        _loadOfflineData: function () {
            if (!this._db) {
                setTimeout(() => this._loadOfflineData(), 100);
                return;
            }
            
            const transaction = this._db.transaction(['projects', 'agents', 'workflows'], 'readonly');
            
            // Load projects
            const projectsRequest = transaction.objectStore('projects').getAll();
            projectsRequest.onsuccess = (event) => {
                this._offlineData.projects = event.target.result;
            };
            
            // Load agents
            const agentsRequest = transaction.objectStore('agents').getAll();
            agentsRequest.onsuccess = (event) => {
                this._offlineData.agents = event.target.result;
            };
            
            // Load workflows
            const workflowsRequest = transaction.objectStore('workflows').getAll();
            workflowsRequest.onsuccess = (event) => {
                this._offlineData.workflows = event.target.result;
            };
            
            transaction.oncomplete = () => {
                // eslint-disable-next-line no-console
                // eslint-disable-next-line no-console
                console.log('Offline data loaded successfully');
                this._fireEvent('offlineDataLoaded', {
                    projectsCount: (this._offlineData.projects || []).length,
                    agentsCount: (this._offlineData.agents || []).length,
                    workflowsCount: (this._offlineData.workflows || []).length
                });
            };
        },

        /**
         * Check storage quota
         */
        _checkStorageQuota: function () {
            if (navigator.storage && navigator.storage.estimate) {
                navigator.storage.estimate().then(estimate => {
                    const usedMB = (estimate.usage / 1024 / 1024).toFixed(2);
                    const quotaMB = (estimate.quota / 1024 / 1024).toFixed(2);
                    const usagePercentage = (estimate.usage / estimate.quota * 100).toFixed(2);
                    
                     
                    
                    // eslint-disable-next-line no-console
                    
                     
                    
                    // eslint-disable-next-line no-console
                    console.log(`Storage usage: ${usedMB} MB / ${quotaMB} MB (${usagePercentage}%)`);
                    
                    if (usagePercentage > 80) {
                        this._showStorageWarning(usagePercentage);
                    }
                });
            }
        },

        /**
         * Show storage warning
         */
        _showStorageWarning: function (usagePercentage) {
            MessageBox.warning(
                `Storage usage is at ${usagePercentage}%. Consider clearing old offline data.`,
                {
                    title: "Storage Warning",
                    actions: [MessageBox.Action.OK, "Clear Old Data"],
                    emphasizedAction: MessageBox.Action.OK,
                    onClose: (sAction) => {
                        if (sAction === "Clear Old Data") {
                            this._clearOldOfflineData();
                        }
                    }
                }
            );
        },

        /* =========================================================== */
        /* Public API                                                  */
        /* =========================================================== */

        /**
         * Get offline data for a specific entity type
         */
        getOfflineData: function (entityType, options = {}) {
            const data = this._offlineData[entityType] || [];
            
            // Apply filters
            let filteredData = data;
            
            if (options.filter) {
                filteredData = data.filter(options.filter);
            }
            
            // Apply sorting
            if (options.sort) {
                filteredData.sort(options.sort);
            }
            
            // Apply pagination
            if (options.skip || options.top) {
                const skip = options.skip || 0;
                const top = options.top || filteredData.length;
                filteredData = filteredData.slice(skip, skip + top);
            }
            
            return filteredData;
        },

        /**
         * Store data for offline use
         */
        storeOfflineData: function (entityType, data, options = {}) {
            if (!this._db) {
                console.warn('IndexedDB not available for offline storage');
                return Promise.reject(new Error('Storage not available'));
            }
            
            return new Promise((resolve, reject) => {
                const transaction = this._db.transaction([entityType], 'readwrite');
                const objectStore = transaction.objectStore(entityType);
                
                // Store metadata
                const metadata = {
                    timestamp: new Date(),
                    source: options.source || 'api',
                    ttl: Date.now() + this._config.offlineTTL
                };
                
                if (Array.isArray(data)) {
                    // Store multiple items
                    data.forEach(item => {
                        const itemWithMetadata = { ...item, _offline: metadata };
                        objectStore.put(itemWithMetadata);
                    });
                } else {
                    // Store single item
                    const itemWithMetadata = { ...data, _offline: metadata };
                    objectStore.put(itemWithMetadata);
                }
                
                transaction.oncomplete = () => {
                    // Update in-memory cache
                    if (!this._offlineData[entityType]) {
                        this._offlineData[entityType] = [];
                    }
                    
                    if (Array.isArray(data)) {
                        this._offlineData[entityType] = data;
                    } else {
                        const index = this._offlineData[entityType].findIndex(item => item.ID === data.ID);
                        if (index >= 0) {
                            this._offlineData[entityType][index] = data;
                        } else {
                            this._offlineData[entityType].push(data);
                        }
                    }
                    
                    resolve();
                };
                
                transaction.onerror = () => {
                    reject(transaction.error);
                };
            });
        },

        /**
         * Queue operation for synchronization
         */
        queueOperation: function (operation) {
            const queueItem = {
                id: this._generateId(),
                operation: operation.type, // 'CREATE', 'UPDATE', 'DELETE'
                entityType: operation.entityType,
                entityId: operation.entityId,
                data: operation.data,
                timestamp: new Date(),
                status: 'pending',
                retries: 0
            };
            
            this._syncQueue.push(queueItem);
            
            // Store in IndexedDB
            this._storeSyncQueueItem(queueItem);
            
            // Try immediate sync if online
            if (this._isOnline) {
                this._processSyncQueue();
            }
            
            return queueItem.id;
        },

        /**
         * Start synchronization
         */
        synchronize: function () {
            if (!this._isOnline) {
                MessageToast.show("Cannot synchronize while offline");
                return Promise.reject(new Error('Offline'));
            }
            
            return this._synchronizeData();
        },

        /**
         * Get network status
         */
        isOnline: function () {
            return this._isOnline;
        },

        /**
         * Get sync queue status
         */
        getSyncStatus: function () {
            return {
                queueLength: this._syncQueue.length,
                syncInProgress: this._syncInProgress,
                lastSyncTime: this._lastSyncTime,
                conflictsCount: this._conflictQueue.length
            };
        },

        /**
         * Clear offline data
         */
        clearOfflineData: function (entityType) {
            if (entityType) {
                return this._clearEntityData(entityType);
            } else {
                return this._clearAllOfflineData();
            }
        },

        /**
         * Handle conflict resolution
         */
        resolveConflict: function (conflictId, resolution) {
            const conflict = this._conflictQueue.find(c => c.id === conflictId);
            if (!conflict) {
                return Promise.reject(new Error('Conflict not found'));
            }
            
            return this._resolveConflict(conflict, resolution);
        },

        /**
         * Get all unresolved conflicts
         * @returns {Promise<Array>} Promise resolving to array of conflicts
         */
        getConflicts: function() {
            return new Promise((resolve) => {
                // Filter only unresolved conflicts from the conflict queue
                const aUnresolvedConflicts = this._conflictQueue.filter(oConflict => 
                    oConflict.status === 'pending' || oConflict.status === 'unresolved'
                );
                resolve(aUnresolvedConflicts);
            });
        },

        /**
         * Skip a conflict (mark as manually resolved)
         * @param {string} sConflictId - Conflict ID
         * @returns {Promise} Promise that resolves when conflict is skipped
         */
        skipConflict: function(sConflictId) {
            return new Promise((resolve, reject) => {
                const iIndex = this._conflictQueue.findIndex(oConflict => oConflict.id === sConflictId);
                if (iIndex === -1) {
                    reject(new Error("Conflict not found"));
                    return;
                }
                
                // Mark conflict as skipped
                this._conflictQueue[iIndex].status = 'skipped';
                this._conflictQueue[iIndex].skippedAt = new Date().toISOString();
                
                // Remove from active conflicts
                this._conflictQueue.splice(iIndex, 1);
                
                resolve();
            });
        },

        /**
         * Get sync queue items
         * @returns {Promise<Array>} Promise resolving to sync queue
         */
        getSyncQueue: function() {
            return new Promise((resolve) => {
                // Return copy of sync queue sorted by timestamp (newest first)
                const aSyncQueue = [...this._syncQueue].sort((a, b) => b.timestamp - a.timestamp);
                resolve(aSyncQueue);
            });
        },

        /* =========================================================== */
        /* Internal synchronization methods                           */
        /* =========================================================== */

        /**
         * Start data synchronization
         */
        _startSynchronization: function () {
            if (this._syncInProgress) {
return;
}
            
            this._synchronizeData();
        },

        /**
         * Synchronize data with server
         */
        _synchronizeData: function () {
            if (this._syncInProgress || !this._isOnline) {
                return Promise.resolve();
            }
            
            this._syncInProgress = true;
            
             
            
            // eslint-disable-next-line no-console
            
             
            
            // eslint-disable-next-line no-console
            console.log('Starting data synchronization...');
            
            return Promise.all([
                this._processSyncQueue(),
                this._downloadFreshData(),
                this._resolveConflicts()
            ])
            .then(() => {
                this._lastSyncTime = new Date();
                this._syncInProgress = false;
                
                 
                
                // eslint-disable-next-line no-console
                
                 
                
                // eslint-disable-next-line no-console
                console.log('Data synchronization completed successfully');
                
                this._fireEvent('syncComplete', {
                    timestamp: this._lastSyncTime,
                    success: true
                });
            })
            .catch(error => {
                this._syncInProgress = false;
                
                console.error('Data synchronization failed:', error);
                
                this._fireEvent('syncError', {
                    timestamp: new Date(),
                    error: error.message
                });
            });
        },

        /**
         * Process sync queue
         */
        _processSyncQueue: function () {
            const pendingItems = this._syncQueue.filter(item => item.status === 'pending');
            
            const syncPromises = pendingItems.map(item => {
                return this._syncItem(item);
            });
            
            return Promise.allSettled(syncPromises);
        },

        /**
         * Sync individual item
         */
        _syncItem: function (item) {
            return new Promise((resolve, reject) => {
                const url = this._buildSyncUrl(item);
                const options = this._buildSyncOptions(item);
                
                fetch(url, options)
                .then(response => {
                    if (response.ok) {
                        item.status = 'completed';
                        this._removeSyncQueueItem(item.id);
                        resolve(item);
                    } else if (response.status === 409) {
                        // Conflict detected
                        return response.json().then(conflictData => {
                            this._handleSyncConflict(item, conflictData);
                            resolve(item);
                        });
                    } else {
                        throw new Error(`Sync failed: ${response.status}`);
                    }
                })
                .catch(error => {
                    item.retries++;
                    if (item.retries >= this._config.maxRetries) {
                        item.status = 'failed';
                        console.error('Sync item failed permanently:', item, error);
                    } else {
                        // Retry later
                        setTimeout(() => {
                            this._syncItem(item);
                        }, this._config.retryInterval * item.retries);
                    }
                    reject(error);
                });
            });
        },

        /**
         * Download fresh data from server
         */
        _downloadFreshData: function () {
            const promises = this._config.criticalDataTypes.map(entityType => {
                return this._downloadEntityData(entityType);
            });
            
            return Promise.allSettled(promises);
        },

        /**
         * Download entity data
         */
        _downloadEntityData: function (entityType) {
            const url = `/api/v1/${entityType}?$top=1000&$orderby=modifiedAt desc`;
            
            return fetch(url, {
                headers: {
                    'Accept': 'application/json',
                    'X-Correlation-Id': this._generateId()
                }
            })
            .then(response => response.json())
            .then(data => {
                return this.storeOfflineData(entityType, data.value || data, {
                    source: 'sync'
                });
            });
        },

        /**
         * Handle sync conflict
         */
        _handleSyncConflict: function (syncItem, conflictData) {
            const conflict = {
                id: this._generateId(),
                syncItemId: syncItem.id,
                entityType: syncItem.entityType,
                entityId: syncItem.entityId,
                clientData: syncItem.data,
                serverData: conflictData.serverData,
                timestamp: new Date(),
                status: 'unresolved'
            };
            
            this._conflictQueue.push(conflict);
            
            // Store conflict in IndexedDB
            this._storeConflict(conflict);
            
            // Auto-resolve based on configuration
            if (this._config.conflictResolution !== 'manual') {
                this._autoResolveConflict(conflict);
            }
        },

        /**
         * Auto-resolve conflict
         */
        _autoResolveConflict: function (conflict) {
            let resolution;
            
            if (this._config.conflictResolution === 'client-wins') {
                resolution = 'useClient';
            } else if (this._config.conflictResolution === 'server-wins') {
                resolution = 'useServer';
            } else {
                // Use timestamp-based resolution
                const clientTime = new Date(conflict.clientData.modifiedAt).getTime();
                const serverTime = new Date(conflict.serverData.modifiedAt).getTime();
                resolution = clientTime > serverTime ? 'useClient' : 'useServer';
            }
            
            this._resolveConflict(conflict, resolution);
        },

        /**
         * Resolve conflict
         */
        _resolveConflict: function (conflict, resolution) {
            return new Promise((resolve, reject) => {
                let dataToUse;
                
                switch (resolution) {
                    case 'useClient':
                        dataToUse = conflict.clientData;
                        break;
                    case 'useServer':
                        dataToUse = conflict.serverData;
                        break;
                    case 'merge':
                        dataToUse = this._mergeConflictData(conflict.clientData, conflict.serverData);
                        break;
                    default:
                        reject(new Error('Invalid resolution type'));
                        return;
                }
                
                // Update server with resolved data
                const url = `/api/v1/${conflict.entityType}/${conflict.entityId}`;
                
                fetch(url, {
                    method: 'PUT',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-Correlation-Id': this._generateId(),
                        'X-Conflict-Resolution': resolution
                    },
                    body: JSON.stringify(dataToUse)
                })
                .then(response => {
                    if (response.ok) {
                        // Update offline data
                        this._updateOfflineEntity(conflict.entityType, conflict.entityId, dataToUse);
                        
                        // Remove conflict
                        this._removeConflict(conflict.id);
                        
                        resolve();
                    } else {
                        throw new Error(`Conflict resolution failed: ${response.status}`);
                    }
                })
                .catch(reject);
            });
        },

        /* =========================================================== */
        /* Utility methods                                            */
        /* =========================================================== */

        /**
         * Generate unique ID
         */
        _generateId: function () {
            return Date.now().toString(36) + Math.random().toString(36).substr(2, 9);
        },

        /**
         * Build sync URL
         */
        _buildSyncUrl: function (item) {
            const baseUrl = `/api/v1/${item.entityType}`;
            
            switch (item.operation) {
                case 'CREATE':
                    return baseUrl;
                case 'UPDATE':
                    return `${baseUrl}/${item.entityId}`;
                case 'DELETE':
                    return `${baseUrl}/${item.entityId}`;
                default:
                    throw new Error('Invalid operation type');
            }
        },

        /**
         * Build sync options
         */
        _buildSyncOptions: function (item) {
            const options = {
                headers: {
                    'Content-Type': 'application/json',
                    'X-Correlation-Id': this._generateId(),
                    'X-Offline-Sync': 'true'
                }
            };
            
            switch (item.operation) {
                case 'CREATE':
                    options.method = 'POST';
                    options.body = JSON.stringify(item.data);
                    break;
                case 'UPDATE':
                    options.method = 'PUT';
                    options.body = JSON.stringify(item.data);
                    break;
                case 'DELETE':
                    options.method = 'DELETE';
                    break;
            }
            
            return options;
        },

        /**
         * Update network status in UI
         */
        _updateNetworkStatus: function (isOnline) {
            // Update global model
            if (sap.ui.getCore().getModel('app')) {
                sap.ui.getCore().getModel('app').setProperty('/networkStatus', {
                    isOnline: isOnline,
                    timestamp: new Date()
                });
            }
            
            // Update document class for CSS styling
            document.body.classList.toggle('offline', !isOnline);
            document.body.classList.toggle('online', isOnline);
        },

        /**
         * Fire custom events
         */
        _fireEvent: function (eventName, data) {
            const event = new CustomEvent(`a2a.offline.${  eventName}`, {
                detail: data
            });
            document.dispatchEvent(event);
        },

        /**
         * Store sync queue item in IndexedDB
         */
        _storeSyncQueueItem: function (item) {
            if (!this._db) {
return;
}
            
            const transaction = this._db.transaction(['syncQueue'], 'readwrite');
            const objectStore = transaction.objectStore('syncQueue');
            objectStore.put(item);
        },

        /**
         * Remove sync queue item
         */
        _removeSyncQueueItem: function (itemId) {
            const index = this._syncQueue.findIndex(item => item.id === itemId);
            if (index >= 0) {
                this._syncQueue.splice(index, 1);
            }
            
            if (!this._db) {
return;
}
            
            const transaction = this._db.transaction(['syncQueue'], 'readwrite');
            const objectStore = transaction.objectStore('syncQueue');
            objectStore.delete(itemId);
        },

        /**
         * Clear old offline data
         */
        _clearOldOfflineData: function () {
            const cutoffTime = Date.now() - this._config.offlineTTL;
            
            if (!this._db) {
return;
}
            
            const transaction = this._db.transaction(['projects', 'agents', 'workflows'], 'readwrite');
            
            ['projects', 'agents', 'workflows'].forEach(storeName => {
                const objectStore = transaction.objectStore(storeName);
                const request = objectStore.openCursor();
                
                request.onsuccess = (event) => {
                    const cursor = event.target.result;
                    if (cursor) {
                        const record = cursor.value;
                        if (record._offline && record._offline.ttl < cutoffTime) {
                            cursor.delete();
                        }
                        cursor.continue();
                    }
                };
            });
            
            transaction.oncomplete = () => {
                MessageToast.show("Old offline data cleared successfully");
                this._loadOfflineData(); // Reload data
            };
        }
    });

    // Create singleton instance
    const oOfflineManager = new OfflineManager();

    return oOfflineManager;
});