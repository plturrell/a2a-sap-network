/**
 * Offline Support Service
 *
 * Provides comprehensive offline capabilities including data synchronization,
 * conflict resolution, and offline queue management for enterprise applications.
 *
 * @author SAP SE
 * @since 1.0.0
 * @version 1.0.0
 */
sap.ui.define([
    "sap/base/Log",
    "./CacheService"
], function(Log, CacheService) {
    "use strict";

    const OfflineService = {

        /* =========================================================== */
        /* Constants                                                   */
        /* =========================================================== */

        OFFLINE_QUEUE_KEY: "a2a-offline-queue",
        SYNC_STATUS_KEY: "a2a-sync-status",
        LAST_SYNC_KEY: "a2a-last-sync",
        CONFLICT_QUEUE_KEY: "a2a-conflicts",

        SYNC_STRATEGIES: {
            CLIENT_WINS: "client-wins",
            SERVER_WINS: "server-wins",
            MERGE: "merge",
            MANUAL: "manual"
        },

        /* =========================================================== */
        /* Lifecycle                                                   */
        /* =========================================================== */

        /**
         * Initialize the offline service
         * @public
         * @since 1.0.0
         */
        init() {
            this._isOnline = navigator.onLine;
            this._syncInProgress = false;
            this._syncQueue = this._loadSyncQueue();
            this._conflictQueue = this._loadConflictQueue();
            this._eventBus = sap.ui.getCore().getEventBus();

            this._registerNetworkEvents();
            this._schedulePeriodicSync();

            // Initialize cache service if not already done
            if (!CacheService._initialized) {
                CacheService.init();
                CacheService._initialized = true;
            }

            Log.info("OfflineService initialized", {
                service: "OfflineService",
                isOnline: this._isOnline,
                queuedOperations: this._syncQueue.length,
                conflicts: this._conflictQueue.length
            });
        },

        /**
         * Destroy the offline service
         * @public
         * @since 1.0.0
         */
        destroy() {
            if (this._syncTimer) {
                clearInterval(this._syncTimer);
            }

            Log.info("OfflineService destroyed", { service: "OfflineService" });
        },

        /* =========================================================== */
        /* Public API                                                  */
        /* =========================================================== */

        /**
         * Check if application is currently online
         * @public
         * @returns {boolean} Online status
         * @since 1.0.0
         */
        isOnline() {
            return this._isOnline;
        },

        /**
         * Execute operation with offline support
         * @public
         * @param {object} operation Operation configuration
         * @param {string} operation.type Operation type (GET, POST, PUT, DELETE)
         * @param {string} operation.url Request URL
         * @param {*} operation.data Request data
         * @param {object} operation.options Request options
         * @param {string} [operation.cacheKey] Cache key for GET requests
         * @param {Array<string>} [operation.tags] Cache tags
         * @param {number} [operation.ttl] Cache TTL
         * @param {string} [operation.syncStrategy] Conflict resolution strategy
         * @returns {Promise} Operation promise
         * @since 1.0.0
         */
        execute(operation) {
            const that = this;

            return new Promise(function(resolve, reject) {
                if (that._isOnline) {
                    // Online - execute immediately
                    that._executeOnline(operation)
                        .then(function(result) {
                            // Cache successful GET responses
                            if (operation.type === "GET" && operation.cacheKey) {
                                CacheService.set(operation.cacheKey, result, {
                                    ttl: operation.ttl,
                                    tags: operation.tags
                                });
                            }
                            resolve(result);
                        })
                        .catch(function(error) {
                            // If it's a GET request, try cache fallback
                            if (operation.type === "GET" && operation.cacheKey) {
                                const cachedData = CacheService.get(operation.cacheKey);
                                if (cachedData) {
                                    Log.debug("Serving cached data due to network error", {
                                        cacheKey: operation.cacheKey
                                    });
                                    resolve(cachedData);
                                    return;
                                }
                            }
                            reject(error);
                        });
                } else {
                    // Offline - handle appropriately
                    if (operation.type === "GET") {
                        // Try to serve from cache
                        if (operation.cacheKey) {
                            const cachedData = CacheService.get(operation.cacheKey);
                            if (cachedData) {
                                Log.debug("Serving cached data (offline)", {
                                    cacheKey: operation.cacheKey
                                });
                                resolve(cachedData);
                                return;
                            }
                        }
                        reject(new Error("No cached data available for offline request"));
                    } else {
                        // Queue write operations for later sync
                        that._queueOperation(operation);
                        Log.info("Operation queued for offline sync", {
                            type: operation.type,
                            url: operation.url
                        });
                        resolve({ queued: true, operationId: operation.id });
                    }
                }
            });
        },

        /**
         * Manually trigger synchronization
         * @public
         * @param {object} options Sync options
         * @param {boolean} [options.force=false] Force sync even if recently synced
         * @param {Array<string>} [options.operations] Specific operation IDs to sync
         * @returns {Promise} Sync promise
         * @since 1.0.0
         */
        sync(options) {
            options = options || {};

            if (this._syncInProgress) {
                return Promise.reject(new Error("Sync already in progress"));
            }

            if (!this._isOnline) {
                return Promise.reject(new Error("Cannot sync while offline"));
            }

            return this._performSync(options);
        },

        /**
         * Get offline queue status
         * @public
         * @returns {object} Queue status
         * @since 1.0.0
         */
        getQueueStatus() {
            return {
                totalOperations: this._syncQueue.length,
                conflicts: this._conflictQueue.length,
                lastSync: this._getLastSyncTime(),
                isOnline: this._isOnline,
                syncInProgress: this._syncInProgress,
                operationsByType: this._getOperationsByType(),
                oldestOperation: this._getOldestOperation()
            };
        },

        /**
         * Clear offline queue
         * @public
         * @param {object} options Clear options
         * @param {boolean} [options.includeConflicts=false] Also clear conflicts
         * @returns {boolean} Success status
         * @since 1.0.0
         */
        clearQueue(options) {
            options = options || {};

            try {
                this._syncQueue = [];
                this._saveSyncQueue();

                if (options.includeConflicts) {
                    this._conflictQueue = [];
                    this._saveConflictQueue();
                }

                this._eventBus.publish("OfflineService", "QueueCleared", {
                    includeConflicts: options.includeConflicts
                });

                Log.info("Offline queue cleared", options);
                return true;
            } catch (error) {
                Log.error("Failed to clear offline queue", { error: error.message });
                return false;
            }
        },

        /**
         * Resolve conflict manually
         * @public
         * @param {string} conflictId Conflict ID
         * @param {object} resolution Resolution data
         * @param {string} resolution.strategy Resolution strategy
         * @param {*} resolution.data Resolved data
         * @returns {Promise} Resolution promise
         * @since 1.0.0
         */
        resolveConflict(conflictId, resolution) {
            const that = this;

            return new Promise(function(resolve, reject) {
                const conflict = that._conflictQueue.find(function(c) {
                    return c.id === conflictId;
                });
                if (!conflict) {
                    reject(new Error("Conflict not found"));
                    return;
                }

                // Apply resolution
                that._applyConflictResolution(conflict, resolution)
                    .then(function(result) {
                        // Remove from conflict queue
                        that._conflictQueue = that._conflictQueue.filter(function(c) {
                            return c.id !== conflictId;
                        });
                        that._saveConflictQueue();

                        that._eventBus.publish("OfflineService", "ConflictResolved", {
                            conflictId,
                            strategy: resolution.strategy
                        });

                        resolve(result);
                    })
                    .catch(reject);
            });
        },

        /* =========================================================== */
        /* Private Methods                                             */
        /* =========================================================== */

        /**
         * Register network status events
         * @private
         * @since 1.0.0
         */
        _registerNetworkEvents() {
            const that = this;

            window.addEventListener("online", function() {
                that._isOnline = true;
                Log.info("Network status changed: online");

                that._eventBus.publish("OfflineService", "NetworkStatusChanged", {
                    isOnline: true
                });

                // Automatically sync when coming back online
                setTimeout(function() {
                    that.sync({ force: false });
                }, 2000); // Wait 2 seconds for network to stabilize
            });

            window.addEventListener("offline", function() {
                that._isOnline = false;
                Log.info("Network status changed: offline");

                that._eventBus.publish("OfflineService", "NetworkStatusChanged", {
                    isOnline: false
                });
            });
        },

        /**
         * Schedule periodic synchronization
         * @private
         * @since 1.0.0
         */
        _schedulePeriodicSync() {
            const that = this;

            // Sync every 5 minutes if online and has queued operations
            this._syncTimer = setInterval(function() {
                if (that._isOnline && that._syncQueue.length > 0 && !that._syncInProgress) {
                    that.sync({ force: false });
                }
            }, 5 * 60 * 1000);
        },

        /**
         * Execute operation online
         * @private
         * @param {object} operation Operation configuration
         * @returns {Promise} Operation promise
         * @since 1.0.0
         */
        _executeOnline(operation) {
            return new Promise(function(resolve, reject) {
                jQuery.ajax({
                    type: operation.type,
                    url: operation.url,
                    data: operation.data ? JSON.stringify(operation.data) : undefined,
                    contentType: "application/json",
                    dataType: "json",
                    timeout: 30000,
                    ...operation.options
                })
                    .done(resolve)
                    .fail(function(xhr, status, error) {
                        reject(new Error(error || status));
                    });
            });
        },

        /**
         * Queue operation for offline sync
         * @private
         * @param {object} operation Operation configuration
         * @since 1.0.0
         */
        _queueOperation(operation) {
            operation.id = this._generateOperationId();
            operation.timestamp = Date.now();
            operation.retryCount = 0;
            operation.status = "queued";

            this._syncQueue.push(operation);
            this._saveSyncQueue();

            this._eventBus.publish("OfflineService", "OperationQueued", {
                operationId: operation.id,
                type: operation.type,
                url: operation.url
            });
        },

        /**
         * Perform synchronization
         * @private
         * @param {object} options Sync options
         * @returns {Promise} Sync promise
         * @since 1.0.0
         */
        _performSync(options) {
            const that = this;

            return new Promise(function(resolve, reject) {
                that._syncInProgress = true;

                that._eventBus.publish("OfflineService", "SyncStarted", {
                    operationCount: that._syncQueue.length
                });

                const syncPromises = [];
                const operationsToSync = options.operations ?
                    that._syncQueue.filter(function(op) {
                        return options.operations.indexOf(op.id) !== -1;
                    }) :
                    that._syncQueue.slice(); // Copy array

                operationsToSync.forEach(function(operation) {
                    syncPromises.push(that._syncOperation(operation));
                });

                Promise.allSettled(syncPromises)
                    .then(function(results) {
                        let successful = 0;
                        let failed = 0;
                        let conflicts = 0;

                        results.forEach(function(result, index) {
                            if (result.status === "fulfilled") {
                                if (result.value.conflict) {
                                    conflicts++;
                                } else {
                                    successful++;
                                    // Remove from queue
                                    that._removeFromQueue(operationsToSync[index].id);
                                }
                            } else {
                                failed++;
                                // Update retry count
                                const operation = operationsToSync[index];
                                operation.retryCount++;
                                operation.lastError = result.reason.message;

                                // Remove if max retries exceeded
                                if (operation.retryCount >= 3) {
                                    that._removeFromQueue(operation.id);
                                }
                            }
                        });

                        that._syncInProgress = false;
                        that._setLastSyncTime();

                        that._eventBus.publish("OfflineService", "SyncCompleted", {
                            successful,
                            failed,
                            conflicts
                        });

                        resolve({
                            successful,
                            failed,
                            conflicts
                        });
                    })
                    .catch(function(error) {
                        that._syncInProgress = false;
                        that._eventBus.publish("OfflineService", "SyncFailed", {
                            error: error.message
                        });
                        reject(error);
                    });
            });
        },

        /**
         * Sync individual operation
         * @private
         * @param {object} operation Operation to sync
         * @returns {Promise} Sync promise
         * @since 1.0.0
         */
        _syncOperation(operation) {
            const that = this;

            return new Promise(function(resolve, reject) {
                that._executeOnline(operation)
                    .then(function(result) {
                        Log.debug("Operation synced successfully", {
                            operationId: operation.id,
                            type: operation.type,
                            url: operation.url
                        });
                        resolve({ success: true, result });
                    })
                    .catch(function(error) {
                        // Check if it's a conflict (409 status)
                        if (error.message.includes("409") || error.message.includes("conflict")) {
                            that._handleConflict(operation, error);
                            resolve({ conflict: true, operation });
                        } else {
                            Log.error("Operation sync failed", {
                                operationId: operation.id,
                                error: error.message
                            });
                            reject(error);
                        }
                    });
            });
        },

        /**
         * Handle sync conflict
         * @private
         * @param {object} operation Conflicted operation
         * @param {Error} error Conflict error
         * @since 1.0.0
         */
        _handleConflict(operation, error) {
            const conflict = {
                id: this._generateConflictId(),
                operationId: operation.id,
                operation,
                error,
                timestamp: Date.now(),
                status: "pending",
                autoResolution: this._getAutoResolutionStrategy(operation)
            };

            this._conflictQueue.push(conflict);
            this._saveConflictQueue();

            this._eventBus.publish("OfflineService", "ConflictDetected", {
                conflictId: conflict.id,
                operationId: operation.id
            });

            // Try auto-resolution if strategy is defined
            if (conflict.autoResolution && conflict.autoResolution !== this.SYNC_STRATEGIES.MANUAL) {
                this._attemptAutoResolution(conflict);
            }
        },

        /**
         * Attempt automatic conflict resolution
         * @private
         * @param {object} conflict Conflict to resolve
         * @since 1.0.0
         */
        _attemptAutoResolution(conflict) {
            const strategy = conflict.autoResolution;
            const resolution = {
                strategy,
                data: this._buildResolutionData(conflict, strategy)
            };

            this.resolveConflict(conflict.id, resolution)
                .then(function() {
                    Log.info("Conflict auto-resolved", {
                        conflictId: conflict.id,
                        strategy
                    });
                })
                .catch(function(error) {
                    Log.error("Auto-resolution failed", {
                        conflictId: conflict.id,
                        error: error.message
                    });
                });
        },

        /**
         * Build resolution data based on strategy
         * @private
         * @param {object} conflict Conflict object
         * @param {string} strategy Resolution strategy
         * @returns {*} Resolution data
         * @since 1.0.0
         */
        _buildResolutionData(conflict, strategy) {
            switch (strategy) {
            case this.SYNC_STRATEGIES.CLIENT_WINS:
                return conflict.operation.data;
            case this.SYNC_STRATEGIES.SERVER_WINS:
                return null; // Will fetch from server
            case this.SYNC_STRATEGIES.MERGE:
                // Simple merge strategy - in production, implement proper merge logic
                return Object.assign({}, conflict.serverData, conflict.operation.data);
            default:
                return null;
            }
        },

        /**
         * Apply conflict resolution
         * @private
         * @param {object} conflict Conflict object
         * @param {object} resolution Resolution data
         * @returns {Promise} Resolution promise
         * @since 1.0.0
         */
        _applyConflictResolution(conflict, resolution) {
            const resolvedOperation = Object.assign({}, conflict.operation);
            resolvedOperation.data = resolution.data;

            return this._executeOnline(resolvedOperation);
        },

        // Storage helper methods

        _loadSyncQueue() {
            try {
                const queueData = localStorage.getItem(this.OFFLINE_QUEUE_KEY);
                return queueData ? JSON.parse(queueData) : [];
            } catch (e) {
                return [];
            }
        },

        _saveSyncQueue() {
            try {
                localStorage.setItem(this.OFFLINE_QUEUE_KEY, JSON.stringify(this._syncQueue));
            } catch (e) {
                Log.error("Failed to save sync queue", { error: e.message });
            }
        },

        _loadConflictQueue() {
            try {
                const conflictData = localStorage.getItem(this.CONFLICT_QUEUE_KEY);
                return conflictData ? JSON.parse(conflictData) : [];
            } catch (e) {
                return [];
            }
        },

        _saveConflictQueue() {
            try {
                localStorage.setItem(this.CONFLICT_QUEUE_KEY, JSON.stringify(this._conflictQueue));
            } catch (e) {
                Log.error("Failed to save conflict queue", { error: e.message });
            }
        },

        _removeFromQueue(operationId) {
            this._syncQueue = this._syncQueue.filter(function(op) {
                return op.id !== operationId;
            });
            this._saveSyncQueue();
        },

        _setLastSyncTime() {
            localStorage.setItem(this.LAST_SYNC_KEY, Date.now().toString());
        },

        _getLastSyncTime() {
            const lastSync = localStorage.getItem(this.LAST_SYNC_KEY);
            return lastSync ? parseInt(lastSync, 10) : null;
        },

        _generateOperationId() {
            return `op-${ Date.now() }-${ Math.random().toString(36).substr(2, 9)}`;
        },

        _generateConflictId() {
            return `conflict-${ Date.now() }-${ Math.random().toString(36).substr(2, 9)}`;
        },

        _getOperationsByType() {
            return this._syncQueue.reduce(function(acc, op) {
                acc[op.type] = (acc[op.type] || 0) + 1;
                return acc;
            }, {});
        },

        _getOldestOperation() {
            if (this._syncQueue.length === 0) {
                return null;
            }
            return this._syncQueue.reduce(function(oldest, op) {
                return op.timestamp < oldest.timestamp ? op : oldest;
            });
        },

        _getAutoResolutionStrategy(operation) {
            // Define auto-resolution strategies based on operation type and URL
            if (operation.type === "PUT" || operation.type === "PATCH") {
                return this.SYNC_STRATEGIES.CLIENT_WINS; // Default for updates
            }
            if (operation.type === "POST") {
                return this.SYNC_STRATEGIES.MERGE; // Try to merge creates
            }
            return this.SYNC_STRATEGIES.MANUAL; // Manual resolution for others
        }
    };

    return OfflineService;
});