sap.ui.define([
    "sap/ui/base/Object",
    "sap/m/MessageToast",
    "sap/m/MessageBox"
], (BaseObject, MessageToast, MessageBox) => {
    "use strict";
/* global localStorage, AbortController, MessageChannel, Intl */

    /**
     * OfflineMixin - Provides offline capabilities for UI5 controllers
     * @namespace com.sap.a2a.controller.mixin.OfflineMixin
     */
    return {
        /**
         * Initialize offline capabilities
         * @public
         */
        initOfflineCapabilities: function () {
            this._oOfflineModel = new sap.ui.model.json.JSONModel();
            this.getView().setModel(this._oOfflineModel, "offline");
            
            this._registerServiceWorker();
            this._initializeOfflineModel();
            this._setupConnectionMonitoring();
            this._loadOfflineSettings();
        },

        /**
         * Register service worker for offline functionality
         * @private
         */
        _registerServiceWorker: function () {
            if ('serviceWorker' in navigator) {
                navigator.serviceWorker.register('./sw.js')
                    .then(registration => {
                        // eslint-disable-next-line no-console
                        // eslint-disable-next-line no-console
                        console.log('[Offline] Service Worker registered:', registration);
                        this._serviceWorkerRegistration = registration;
                        
                        // Listen for service worker updates
                        registration.addEventListener('updatefound', () => {
                            const newWorker = registration.installing;
                            newWorker.addEventListener('statechange', () => {
                                if (newWorker.state === 'installed' && navigator.serviceWorker.controller) {
                                    this._showUpdateAvailableMessage();
                                }
                            });
                        });
                    })
                    .catch(error => {
                        console.error('[Offline] Service Worker registration failed:', error);
                    });

                // Listen for service worker messages
                navigator.serviceWorker.addEventListener('message', this._handleServiceWorkerMessage.bind(this));
            } else {
                console.warn('[Offline] Service Workers not supported');
                this._oOfflineModel.setProperty("/serviceWorkerSupported", false);
            }
        },

        /**
         * Initialize offline model with default data
         * @private
         */
        _initializeOfflineModel: function () {
            const oOfflineData = {
                isOnline: navigator.onLine,
                isOfflineModeEnabled: true,
                serviceWorkerSupported: 'serviceWorker' in navigator,
                connectionIcon: navigator.onLine ? "sap-icon://connected" : "sap-icon://disconnected",
                connectionColor: navigator.onLine ? "Positive" : "Critical",
                connectionText: navigator.onLine ? "Connected" : "Offline",
                isReconnecting: false,
                lastSyncTime: this._formatDate(new Date()),
                cacheSize: "Calculating...",
                queuedRequests: [],
                availableFeatures: [
                    {
                        id: "viewProjects",
                        title: "View Projects",
                        description: "Browse cached project data",
                        icon: "sap-icon://folder-blank"
                    },
                    {
                        id: "viewAgents",
                        title: "View Agents",
                        description: "Check agent status and information",
                        icon: "sap-icon://group"
                    },
                    {
                        id: "viewDashboard",
                        title: "Dashboard",
                        description: "View cached analytics and metrics",
                        icon: "sap-icon://business-objects-experience"
                    },
                    {
                        id: "offlineSettings",
                        title: "Settings",
                        description: "Configure offline preferences",
                        icon: "sap-icon://action-settings"
                    }
                ],
                helpItems: [
                    {
                        title: "Data Synchronization",
                        description: "Changes made offline will sync when connection is restored",
                        icon: "sap-icon://synchronize"
                    },
                    {
                        title: "Limited Functionality",
                        description: "Some features require an active internet connection",
                        icon: "sap-icon://information"
                    },
                    {
                        title: "Cache Management",
                        description: "Cached data is automatically cleaned based on your settings",
                        icon: "sap-icon://database"
                    }
                ],
                settings: {
                    enableOffline: true,
                    autoSync: true,
                    cacheTimeout: 86400, // 24 hours
                    maxCacheSize: 50 // MB
                }
            };

            this._oOfflineModel.setData(oOfflineData);
            this._updateCacheStatus();
        },

        /**
         * Setup connection monitoring
         * @private
         */
        _setupConnectionMonitoring: function () {
            window.addEventListener('online', this._handleOnline.bind(this));
            window.addEventListener('offline', this._handleOffline.bind(this));

            // Periodic connection check
            this._connectionCheckInterval = setInterval(() => {
                this._checkConnectionStatus();
            }, 30000); // Check every 30 seconds
        },

        /**
         * Handle online event
         * @private
         */
        _handleOnline: function () {
            // eslint-disable-next-line no-console
            // eslint-disable-next-line no-console
            console.log('[Offline] Connection restored');
            
            this._oOfflineModel.setProperty("/isOnline", true);
            this._oOfflineModel.setProperty("/connectionIcon", "sap-icon://connected");
            this._oOfflineModel.setProperty("/connectionColor", "Positive");
            this._oOfflineModel.setProperty("/connectionText", "Connected");
            this._oOfflineModel.setProperty("/isReconnecting", false);

            MessageToast.show("Connection restored. Syncing data...");
            
            // Trigger background sync
            this._triggerBackgroundSync();
            
            // Update cache status
            this._updateCacheStatus();
        },

        /**
         * Handle offline event
         * @private
         */
        _handleOffline: function () {
            // eslint-disable-next-line no-console
            // eslint-disable-next-line no-console
            console.log('[Offline] Connection lost');
            
            this._oOfflineModel.setProperty("/isOnline", false);
            this._oOfflineModel.setProperty("/connectionIcon", "sap-icon://disconnected");
            this._oOfflineModel.setProperty("/connectionColor", "Critical");
            this._oOfflineModel.setProperty("/connectionText", "Offline");

            MessageToast.show("Connection lost. Working offline...");
            
            // Navigate to offline page if not already there
            if (this.getRouter && this.getRouter().getHashChanger().getHash() !== "offline") {
                setTimeout(() => {
                    MessageBox.information("You are now offline. Some features may be limited.", {
                        title: "Offline Mode",
                        actions: [MessageBox.Action.OK, "Go to Offline Page"],
                        emphasizedAction: MessageBox.Action.OK,
                        onClose: (sAction) => {
                            if (sAction === "Go to Offline Page") {
                                this.getRouter().navTo("offline");
                            }
                        }
                    });
                }, 2000);
            }
        },

        /**
         * Check connection status with server ping
         * @private
         */
        _checkConnectionStatus: function () {
            if (!navigator.onLine) {
return;
}

            // Ping server to verify actual connectivity
            const pingUrl = `${window.location.origin}/api/v1/health`;
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000);

            fetch(pingUrl, {
                method: 'HEAD',
                signal: controller.signal,
                cache: 'no-cache'
            })
            .then(response => {
                clearTimeout(timeoutId);
                if (response.ok && !this._oOfflineModel.getProperty("/isOnline")) {
                    this._handleOnline();
                }
            })
            .catch(() => {
                clearTimeout(timeoutId);
                if (this._oOfflineModel.getProperty("/isOnline")) {
                    this._handleOffline();
                }
            });
        },

        /**
         * Handle service worker messages
         * @private
         * @param {MessageEvent} event Message event
         */
        _handleServiceWorkerMessage: function (event) {
            const { data } = event;
            
            if (data.type === 'CACHE_UPDATED') {
                this._updateCacheStatus();
            } else if (data.type === 'SYNC_COMPLETE') {
                MessageToast.show("Data synchronized successfully");
                this._oOfflineModel.setProperty("/lastSyncTime", this._formatDate(new Date()));
            } else if (data.type === 'SYNC_ERROR') {
                MessageToast.show("Synchronization failed. Will retry later.");
            }
        },

        /**
         * Update cache status information
         * @private
         */
        _updateCacheStatus: function () {
            if (!navigator.serviceWorker.controller) {
return;
}

            const messageChannel = new MessageChannel();
            messageChannel.port1.onmessage = (event) => {
                const { success, caches, error } = event.data;
                
                if (success) {
                    let totalSize = 0;
                    let _totalItems = 0;
                    
                    Object.values(caches).forEach(cache => {
                        _totalItems += cache.size;
                        // Estimate size (rough calculation)
                        totalSize += cache.size * 5; // 5KB average per cached item
                    });
                    
                    this._oOfflineModel.setProperty("/cacheSize", this._formatBytes(totalSize * 1024));
                } else {
                    console.error('[Offline] Failed to get cache status:', error);
                    this._oOfflineModel.setProperty("/cacheSize", "Unknown");
                }
            };

            navigator.serviceWorker.controller.postMessage(
                { type: 'GET_CACHE_STATUS' },
                [messageChannel.port2]
            );
        },

        /**
         * Trigger background sync
         * @private
         */
        _triggerBackgroundSync: function () {
            if ('serviceWorker' in navigator && 'sync' in window.ServiceWorkerRegistration.prototype) {
                navigator.serviceWorker.ready.then(registration => {
                    return registration.sync.register('background-sync');
                });
            }
        },

        /**
         * Load offline settings from storage
         * @private
         */
        _loadOfflineSettings: function () {
            const savedSettings = localStorage.getItem('a2a-offline-settings');
            if (savedSettings) {
                try {
                    const settings = JSON.parse(savedSettings);
                    this._oOfflineModel.setProperty("/settings", {
                        ...this._oOfflineModel.getProperty("/settings"),
                        ...settings
                    });
                } catch (error) {
                    console.error('[Offline] Failed to load settings:', error);
                }
            }
        },

        /**
         * Save offline settings to storage
         * @private
         */
        _saveOfflineSettings: function () {
            const settings = this._oOfflineModel.getProperty("/settings");
            localStorage.setItem('a2a-offline-settings', JSON.stringify(settings));
        },

        /**
         * Show update available message
         * @private
         */
        _showUpdateAvailableMessage: function () {
            MessageBox.information(
                "A new version of the application is available. Please refresh to update.",
                {
                    title: "Update Available",
                    actions: [MessageBox.Action.OK, "Refresh Now"],
                    emphasizedAction: "Refresh Now",
                    onClose: (sAction) => {
                        if (sAction === "Refresh Now") {
                            window.location.reload();
                        }
                    }
                }
            );
        },

        /**
         * Handle retry connection button press
         * @public
         */
        onRetryConnection: function () {
            this._oOfflineModel.setProperty("/isReconnecting", true);
            
            setTimeout(() => {
                this._checkConnectionStatus();
                this._oOfflineModel.setProperty("/isReconnecting", false);
            }, 2000);
        },

        /**
         * Handle offline feature press
         * @public
         * @param {sap.ui.base.Event} oEvent Press event
         */
        onOfflineFeaturePress: function (oEvent) {
            const sFeatureId = oEvent.getSource().data("featureId");
            
            switch (sFeatureId) {
                case "viewProjects":
                    this.getRouter().navTo("ProjectsList");
                    break;
                case "viewAgents":
                    this.getRouter().navTo("AgentsList");
                    break;
                case "viewDashboard":
                    this.getRouter().navTo("dashboard");
                    break;
                case "offlineSettings":
                    // Settings are on the same page
                    break;
                default:
                    MessageToast.show("Feature not available offline");
            }
        },

        /**
         * Handle clear cache button press
         * @public
         */
        onClearCache: function () {
            MessageBox.warning(
                "This will clear all cached data. You may need to reload data when online.",
                {
                    title: "Clear Cache",
                    actions: [MessageBox.Action.YES, MessageBox.Action.NO],
                    emphasizedAction: MessageBox.Action.NO,
                    onClose: (sAction) => {
                        if (sAction === MessageBox.Action.YES) {
                            this._clearCache();
                        }
                    }
                }
            );
        },

        /**
         * Clear all cache data
         * @private
         */
        _clearCache: function () {
            if (!navigator.serviceWorker.controller) {
return;
}

            const messageChannel = new MessageChannel();
            messageChannel.port1.onmessage = (event) => {
                const { success } = event.data;
                if (success) {
                    MessageToast.show("Cache cleared successfully");
                    this._updateCacheStatus();
                } else {
                    MessageToast.show("Failed to clear cache");
                }
            };

            navigator.serviceWorker.controller.postMessage(
                { type: 'CLEAR_CACHE' },
                [messageChannel.port2]
            );
        },

        /**
         * Handle export offline data button press
         * @public
         */
        onExportOfflineData: function () {
            // Implementation for exporting cached data
            MessageToast.show("Export functionality coming soon");
        },

        /**
         * Handle process sync queue button press
         * @public
         */
        onProcessSyncQueue: function () {
            this._triggerBackgroundSync();
            MessageToast.show("Processing sync queue...");
        },

        /**
         * Handle offline mode toggle
         * @public
         * @param {sap.ui.base.Event} oEvent Switch event
         */
        onOfflineModeToggle: function (oEvent) {
            const bEnabled = oEvent.getParameter("state");
            this._oOfflineModel.setProperty("/settings/enableOffline", bEnabled);
            this._saveOfflineSettings();
            
            if (bEnabled) {
                MessageToast.show("Offline mode enabled");
            } else {
                MessageToast.show("Offline mode disabled");
            }
        },

        /**
         * Handle auto sync toggle
         * @public
         * @param {sap.ui.base.Event} oEvent Switch event
         */
        onAutoSyncToggle: function (oEvent) {
            const bEnabled = oEvent.getParameter("state");
            this._oOfflineModel.setProperty("/settings/autoSync", bEnabled);
            this._saveOfflineSettings();
        },

        /**
         * Handle cache timeout change
         * @public
         * @param {sap.ui.base.Event} oEvent Selection change event
         */
        onCacheTimeoutChange: function (oEvent) {
            const sValue = oEvent.getParameter("selectedItem").getKey();
            this._oOfflineModel.setProperty("/settings/cacheTimeout", parseInt(sValue));
            this._saveOfflineSettings();
        },

        /**
         * Handle max cache size change
         * @public
         * @param {sap.ui.base.Event} oEvent Selection change event
         */
        onMaxCacheSizeChange: function (oEvent) {
            const sValue = oEvent.getParameter("selectedItem").getKey();
            this._oOfflineModel.setProperty("/settings/maxCacheSize", parseInt(sValue));
            this._saveOfflineSettings();
        },

        /**
         * Handle back to app button press
         * @public
         */
        onBackToApp: function () {
            this.getRouter().navTo("ProjectsList");
        },

        /**
         * Format bytes to human readable format
         * @private
         * @param {number} bytes Number of bytes
         * @returns {string} Formatted string
         */
        _formatBytes: function (bytes) {
            if (bytes === 0) {
return '0 Bytes';
}
            
            const k = 1024;
            const sizes = ['Bytes', 'KB', 'MB', 'GB'];
            const i = Math.floor(Math.log(bytes) / Math.log(k));
            
            return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))  } ${  sizes[i]}`;
        },

        /**
         * Format date to readable string
         * @private
         * @param {Date} date Date to format
         * @returns {string} Formatted date string
         */
        _formatDate: function (date) {
            return new Intl.DateTimeFormat('en-US', {
                year: 'numeric',
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            }).format(date);
        },

        /**
         * Cleanup offline resources
         * @public
         */
        cleanupOfflineCapabilities: function () {
            if (this._connectionCheckInterval) {
                clearInterval(this._connectionCheckInterval);
            }
            
            window.removeEventListener('online', this._handleOnline);
            window.removeEventListener('offline', this._handleOffline);
        }
    };
});