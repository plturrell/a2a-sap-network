sap.ui.define([
    "sap/ui/base/Object",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast"
], (BaseObject, JSONModel, MessageToast) => {
    "use strict";
/* global  */

    /**
     * A2A Agents - Notification Service
     * Handles notification data management and API communication for the UI5 frontend.
     * @class a2a.portal.services.NotificationService
     * @extends sap.ui.base.Object
     * @description Comprehensive notification management service that handles loading, filtering,
     * updating, and real-time synchronization of notifications. Supports various notification
     * types, priorities, and user actions.
     * 
     * @example
     * // Basic initialization in controller
     * onInit: function() {
     *     this._notificationService = new NotificationService();
     *     this.getView().setModel(
     *         this._notificationService.getModel(),
     *         "notifications"
     *     );
     *     this._notificationService.loadNotifications(true);
     * }
     * 
     * @example
     * // Bind to UI elements
     * <NotificationList
     *     items="{notifications>/notifications}"
     *     noDataText="No notifications"
     *     showNoData="true">
     *     <NotificationListItem
     *         title="{notifications>title}"
     *         description="{notifications>message}"
     *         unread="{= ${notifications>status} === 'unread'}"
     *         priority="{notifications>priority}"
     *         datetime="{notifications>created_at}"
     *     />
     * </NotificationList>
     * 
     * @example
     * // Advanced usage with filtering and actions
     * const service = new NotificationService();
     * 
     * // Set filters
     * service.setFilters({
     *     status: "unread",
     *     type: "agent"
     * });
     * 
     * // Handle notification actions
     * service.getModel().attachPropertyChange(function(oEvent) {
     *     if (oEvent.getParameter("path").includes("/notifications/")) {
     *         const notification = oEvent.getParameter("context").getObject();
     *         this.handleNotificationChange(notification);
     *     }
     * }.bind(this));
     */
    return BaseObject.extend("a2a.portal.services.NotificationService", {

        /**
         * Constructor
         * @description Initializes the notification service with default model structure,
         * sets up auto-refresh, and prepares the service for use.
         * 
         * @example
         * // Create with custom configuration
         * const service = new NotificationService();
         * 
         * // Override default settings
         * service._notificationModel.setProperty("/currentLimit", 50);
         * service.stopAutoRefresh(); // Disable auto-refresh
         * 
         * @example  
         * // Create with immediate load
         * const service = new NotificationService();
         * service.loadNotifications(true).then(() => {
         // eslint-disable-next-line no-console
         *     console.log("Service ready with initial data");
         * });
         */
        constructor: function () {
            // eslint-disable-next-line prefer-rest-params
            BaseObject.prototype.constructor.apply(this, arguments);
            
            // Initialize notification model
            this._notificationModel = new JSONModel({
                notifications: [],
                stats: {
                    total: 0,
                    unread: 0,
                    read: 0,
                    dismissed: 0,
                    critical: 0,
                    high: 0,
                    medium: 0,
                    low: 0
                },
                loading: false,
                hasMore: false,
                currentOffset: 0,
                currentLimit: 20,
                filters: {
                    status: null,
                    type: null
                }
            });
            
            // Auto-refresh interval (5 minutes)
            this._refreshInterval = null;
            this._startAutoRefresh();
        },

        /**
         * Get the notification model
         * @returns {sap.ui.model.json.JSONModel} The notification model containing all notification data
         * 
         * @example
         * // Bind to view
         * this.getView().setModel(
         *     this._notificationService.getModel(),
         *     "notifications"
         * );
         * 
         * @example
         * // Access model data directly
         * const unreadCount = this._notificationService.getModel()
         *     .getProperty("/stats/unread");
         * 
         * @example
         * // Listen for changes
         * this._notificationService.getModel().attachPropertyChange(
         *     function(oEvent) {
         *         const path = oEvent.getParameter("path");
         *         const value = oEvent.getParameter("value");
         // eslint-disable-next-line no-console
         *         console.log(`Property ${path} changed to ${value}`);
         *     }
         * );
         */
        getModel: function () {
            return this._notificationModel;
        },

        /**
         * Load notifications from the backend
         * @param {boolean} bReset - Whether to reset the current notifications
         * @returns {Promise} Promise that resolves when notifications are loaded
         */
        loadNotifications: function (bReset) {
            const _that = this;
            const oModel = this._notificationModel;
            const oData = oModel.getData();
            
            if (bReset) {
                oData.currentOffset = 0;
                oData.notifications = [];
            }
            
            oData.loading = true;
            oModel.setData(oData);
            
            // Build query parameters
            const aParams = [];
            aParams.push(`limit=${  oData.currentLimit}`);
            aParams.push(`offset=${  oData.currentOffset}`);
            
            if (oData.filters.status) {
                aParams.push(`status=${  oData.filters.status}`);
            }
            
            if (oData.filters.type) {
                aParams.push(`type=${  oData.filters.type}`);
            }
            
            const sUrl = `/api/notifications?${  aParams.join("&")}`;
            
            return new Promise((resolve, reject) => {
                jQuery.ajax({
                    url: sUrl,
                    method: "GET",
                    success: function (oResponse) {
                        const aNewNotifications = oResponse.notifications || [];
                        
                        if (bReset) {
                            oData.notifications = aNewNotifications;
                        } else {
                            oData.notifications = oData.notifications.concat(aNewNotifications);
                        }
                        
                        oData.stats.total = oResponse.total || 0;
                        oData.stats.unread = oResponse.unread_count || 0;
                        oData.hasMore = oResponse.has_more || false;
                        oData.loading = false;
                        
                        oModel.setData(oData);
                        
                         
                        
                        // eslint-disable-next-line no-console
                        
                         
                        
                        // eslint-disable-next-line no-console
                        console.log(`Loaded ${  aNewNotifications.length  } notifications`);
                        resolve(oResponse);
                    },
                    error: function (oError) {
                        console.error("Failed to load notifications:", oError);
                        oData.loading = false;
                        oModel.setData(oData);
                        
                        MessageToast.show("Failed to load notifications");
                        reject(oError);
                    }
                });
            });
        },

        /**
         * Load more notifications (pagination)
         * @returns {Promise} Promise that resolves when more notifications are loaded
         */
        loadMoreNotifications: function () {
            const oData = this._notificationModel.getData();
            
            if (!oData.hasMore || oData.loading) {
                return Promise.resolve();
            }
            
            oData.currentOffset += oData.currentLimit;
            this._notificationModel.setData(oData);
            
            return this.loadNotifications(false);
        },

        /**
         * Load notification statistics
         * @returns {Promise} Promise that resolves when stats are loaded
         */
        loadStats: function () {
            const _that = this;
            const oModel = this._notificationModel;
            
            return new Promise((resolve, reject) => {
                jQuery.ajax({
                    url: "/api/notifications/stats",
                    method: "GET",
                    success: function (oResponse) {
                        const oData = oModel.getData();
                        oData.stats = oResponse;
                        oModel.setData(oData);
                        
                         
                        
                        // eslint-disable-next-line no-console
                        
                         
                        
                        // eslint-disable-next-line no-console
                        console.log("Loaded notification stats:", oResponse);
                        resolve(oResponse);
                    },
                    error: function (oError) {
                        console.error("Failed to load notification stats:", oError);
                        reject(oError);
                    }
                });
            });
        },

        /**
         * Mark a notification as read
         * @param {string} sNotificationId - The notification ID
         * @returns {Promise} Promise that resolves when notification is marked as read
         */
        markAsRead: function (sNotificationId) {
            const _that = this;
            const oModel = this._notificationModel;
            
            return new Promise((resolve, reject) => {
                jQuery.ajax({
                    url: `/api/notifications/${  sNotificationId  }/read`,
                    method: "PATCH",
                    success: function (oResponse) {
                        // Update the notification in the model
                        const oData = oModel.getData();
                        const oNotification = oData.notifications.find((n) => {
                            return n.id === sNotificationId;
                        });
                        
                        if (oNotification && oNotification.status === "unread") {
                            oNotification.status = "read";
                            oNotification.read_at = new Date().toISOString();
                            oData.stats.unread = Math.max(0, oData.stats.unread - 1);
                            oData.stats.read += 1;
                            oModel.setData(oData);
                        }
                        
                         
                        
                        // eslint-disable-next-line no-console
                        
                         
                        
                        // eslint-disable-next-line no-console
                        console.log("Marked notification as read:", sNotificationId);
                        resolve(oResponse);
                    },
                    error: function (oError) {
                        console.error("Failed to mark notification as read:", oError);
                        MessageToast.show("Failed to mark notification as read");
                        reject(oError);
                    }
                });
            });
        },

        /**
         * Dismiss a notification
         * @param {string} sNotificationId - The notification ID
         * @returns {Promise} Promise that resolves when notification is dismissed
         */
        dismissNotification: function (sNotificationId) {
            const _that = this;
            const oModel = this._notificationModel;
            
            return new Promise((resolve, reject) => {
                jQuery.ajax({
                    url: `/api/notifications/${  sNotificationId  }/dismiss`,
                    method: "PATCH",
                    success: function (oResponse) {
                        // Update the notification in the model
                        const oData = oModel.getData();
                        const oNotification = oData.notifications.find((n) => {
                            return n.id === sNotificationId;
                        });
                        
                        if (oNotification) {
                            oNotification.status = "dismissed";
                            oNotification.dismissed_at = new Date().toISOString();
                            
                            if (oNotification.status === "unread") {
                                oData.stats.unread = Math.max(0, oData.stats.unread - 1);
                            }
                            oData.stats.dismissed += 1;
                            oModel.setData(oData);
                        }
                        
                         
                        
                        // eslint-disable-next-line no-console
                        
                         
                        
                        // eslint-disable-next-line no-console
                        console.log("Dismissed notification:", sNotificationId);
                        MessageToast.show("Notification dismissed");
                        resolve(oResponse);
                    },
                    error: function (oError) {
                        console.error("Failed to dismiss notification:", oError);
                        MessageToast.show("Failed to dismiss notification");
                        reject(oError);
                    }
                });
            });
        },

        /**
         * Mark all notifications as read
         * @returns {Promise} Promise that resolves when all notifications are marked as read
         */
        markAllAsRead: function () {
            const _that = this;
            const oModel = this._notificationModel;
            
            return new Promise((resolve, reject) => {
                jQuery.ajax({
                    url: "/api/notifications/mark-all-read",
                    method: "POST",
                    success: function (oResponse) {
                        // Update all unread notifications in the model
                        const oData = oModel.getData();
                        const now = new Date().toISOString();
                        
                        oData.notifications.forEach((oNotification) => {
                            if (oNotification.status === "unread") {
                                oNotification.status = "read";
                                oNotification.read_at = now;
                            }
                        });
                        
                        oData.stats.read += oData.stats.unread;
                        oData.stats.unread = 0;
                        oModel.setData(oData);
                        
                         
                        
                        // eslint-disable-next-line no-console
                        
                         
                        
                        // eslint-disable-next-line no-console
                        console.log("Marked all notifications as read");
                        MessageToast.show("All notifications marked as read");
                        resolve(oResponse);
                    },
                    error: function (oError) {
                        console.error("Failed to mark all notifications as read:", oError);
                        MessageToast.show("Failed to mark all notifications as read");
                        reject(oError);
                    }
                });
            });
        },

        /**
         * Delete a notification
         * @param {string} sNotificationId - The notification ID
         * @returns {Promise} Promise that resolves when notification is deleted
         */
        deleteNotification: function (sNotificationId) {
            const _that = this;
            const oModel = this._notificationModel;
            
            return new Promise((resolve, reject) => {
                jQuery.ajax({
                    url: `/api/notifications/${  sNotificationId}`,
                    method: "DELETE",
                    success: function (oResponse) {
                        // Remove the notification from the model
                        const oData = oModel.getData();
                        const iIndex = oData.notifications.findIndex((n) => {
                            return n.id === sNotificationId;
                        });
                        
                        if (iIndex >= 0) {
                            const oNotification = oData.notifications[iIndex];
                            oData.notifications.splice(iIndex, 1);
                            
                            // Update stats
                            oData.stats.total = Math.max(0, oData.stats.total - 1);
                            if (oNotification.status === "unread") {
                                oData.stats.unread = Math.max(0, oData.stats.unread - 1);
                            } else if (oNotification.status === "read") {
                                oData.stats.read = Math.max(0, oData.stats.read - 1);
                            } else if (oNotification.status === "dismissed") {
                                oData.stats.dismissed = Math.max(0, oData.stats.dismissed - 1);
                            }
                            
                            oModel.setData(oData);
                        }
                        
                         
                        
                        // eslint-disable-next-line no-console
                        
                         
                        
                        // eslint-disable-next-line no-console
                        console.log("Deleted notification:", sNotificationId);
                        MessageToast.show("Notification deleted");
                        resolve(oResponse);
                    },
                    error: function (oError) {
                        console.error("Failed to delete notification:", oError);
                        MessageToast.show("Failed to delete notification");
                        reject(oError);
                    }
                });
            });
        },

        /**
         * Set notification filters
         * @param {object} oFilters - Filter object with status and type properties
         */
        setFilters: function (oFilters) {
            const oData = this._notificationModel.getData();
            oData.filters = oFilters || { status: null, type: null };
            this._notificationModel.setData(oData);
            
            // Reload notifications with new filters
            this.loadNotifications(true);
        },

        /**
         * Get the unread notification count
         * @returns {number} The number of unread notifications
         */
        getUnreadCount: function () {
            return this._notificationModel.getProperty("/stats/unread") || 0;
        },

        /**
         * Start auto-refresh of notifications
         * @private
         */
        _startAutoRefresh: function () {
            const _that = this;
            
            // Clear existing interval
            if (this._refreshInterval) {
                clearInterval(this._refreshInterval);
            }
            
            // Set up new interval (5 minutes)
            this._refreshInterval = setInterval(() => {
                _that.loadStats();
                // Only refresh if we're on the first page
                const oData = _that._notificationModel.getData();
                if (oData.currentOffset === 0) {
                    _that.loadNotifications(true);
                }
            }, 300000); // 5 minutes
        },

        /**
         * Stop auto-refresh
         */
        stopAutoRefresh: function () {
            if (this._refreshInterval) {
                clearInterval(this._refreshInterval);
                this._refreshInterval = null;
            }
        },

        /**
         * Destroy the service and clean up resources
         */
        destroy: function () {
            this.stopAutoRefresh();
            // eslint-disable-next-line prefer-rest-params
            BaseObject.prototype.destroy.apply(this, arguments);
        }
    });
});
