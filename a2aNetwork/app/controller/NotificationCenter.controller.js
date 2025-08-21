sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/m/MessageToast",
    "sap/m/MessageBox",
    "sap/ui/core/Fragment"
], function (Controller, JSONModel, MessageToast, MessageBox, Fragment) {
    "use strict";

    return Controller.extend("a2a.network.controller.NotificationCenter", {
        
        onInit: function () {
            // Initialize models
            this.initializeModels();
            
            // Initialize WebSocket connection
            this.initializeWebSocket();
            
            // Load initial notifications
            this.loadNotifications();
            
            // Set up auto-refresh
            this.setupAutoRefresh();
        },

        initializeModels: function () {
            // Notification data model
            this.oNotificationModel = new JSONModel({
                notifications: [],
                stats: {
                    total: 0,
                    unread: 0,
                    byType: {},
                    byPriority: {},
                    byStatus: {}
                },
                connectionStatus: 'disconnected'
            });
            this.getView().setModel(this.oNotificationModel, "notificationModel");

            // Filter model
            this.oFilterModel = new JSONModel({
                status: "",
                type: "",
                priority: "",
                category: ""
            });
            this.getView().setModel(this.oFilterModel, "filterModel");

            // Pagination model
            this.oPaginationModel = new JSONModel({
                limit: 20,
                offset: 0,
                currentPage: 1,
                totalPages: 1,
                total: 0
            });
            this.getView().setModel(this.oPaginationModel, "paginationModel");

            // Settings model (loaded separately)
            this.oSettingsModel = new JSONModel();
            this.getView().setModel(this.oSettingsModel, "settingsModel");
        },

        initializeWebSocket: function () {
            const wsUrl = this.getWebSocketUrl();
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = this.onWebSocketOpen.bind(this);
            this.ws.onmessage = this.onWebSocketMessage.bind(this);
            this.ws.onclose = this.onWebSocketClose.bind(this);
            this.ws.onerror = this.onWebSocketError.bind(this);

            // Store reconnection parameters
            this.wsUrl = wsUrl;
            this.reconnectAttempts = 0;
            this.maxReconnectAttempts = 5;
            this.reconnectDelay = 1000;
        },

        getWebSocketUrl: function () {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            const host = location.host;
            return `${protocol}//${host}/notifications/v2`;
        },

        onWebSocketOpen: function () {
            console.log("WebSocket connected");
            this.oNotificationModel.setProperty("/connectionStatus", "connected");
            this.reconnectAttempts = 0;
            
            // Authenticate
            const userId = this.getCurrentUserId();
            this.sendWebSocketMessage({
                type: 'auth',
                userId: userId
            });
        },

        onWebSocketMessage: function (event) {
            try {
                const message = JSON.parse(event.data);
                this.handleWebSocketMessage(message);
            } catch (error) {
                console.error("Failed to parse WebSocket message:", error);
            }
        },

        onWebSocketClose: function (event) {
            console.log("WebSocket disconnected:", event.code, event.reason);
            this.oNotificationModel.setProperty("/connectionStatus", "disconnected");
            
            // Attempt reconnection
            this.attemptReconnection();
        },

        onWebSocketError: function (error) {
            console.error("WebSocket error:", error);
            this.oNotificationModel.setProperty("/connectionStatus", "error");
        },

        attemptReconnection: function () {
            if (this.reconnectAttempts >= this.maxReconnectAttempts) {
                console.error("Max reconnection attempts reached");
                MessageToast.show("Connection lost. Please refresh the page.");
                return;
            }

            this.reconnectAttempts++;
            const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
            
            console.log(`Attempting reconnection ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${delay}ms`);
            
            setTimeout(() => {
                this.initializeWebSocket();
            }, delay);
        },

        handleWebSocketMessage: function (message) {
            switch (message.type) {
                case 'connection':
                    this.clientId = message.clientId;
                    this.reconnectToken = message.reconnectToken;
                    break;
                    
                case 'auth_success':
                    console.log("Authentication successful");
                    this.updateNotifications(message.notifications);
                    this.updateStats(message);
                    this.loadUserPreferences();
                    break;
                    
                case 'new_notification':
                    this.addNotification(message.notification);
                    this.showNotificationToast(message.notification);
                    break;
                    
                case 'notification_read':
                    this.markNotificationAsRead(message.notificationId);
                    break;
                    
                case 'all_notifications_read':
                    this.markAllNotificationsAsRead();
                    break;
                    
                case 'notification_dismissed':
                    this.removeNotification(message.notificationId);
                    break;
                    
                case 'notifications':
                    this.updateNotifications(message.notifications);
                    this.updateStats(message.stats);
                    break;
                    
                case 'preferences_updated':
                    this.oSettingsModel.setData(message.preferences);
                    MessageToast.show("Preferences updated successfully");
                    break;
                    
                case 'error':
                    console.error("WebSocket error:", message.message);
                    MessageToast.show("Error: " + message.message);
                    break;
                    
                case 'pong':
                    // Handle ping response
                    break;
            }
        },

        sendWebSocketMessage: function (message) {
            if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                this.ws.send(JSON.stringify(message));
            } else {
                console.warn("WebSocket not connected, message queued");
                // TODO: Implement message queuing
            }
        },

        getCurrentUserId: function () {
            // Get current user ID from session or context
            // This is a placeholder - implement based on your authentication system
            return sap.ushell?.Container?.getUser?.()?.getId?.() || "demo-user";
        },

        loadNotifications: function () {
            const filters = this.oFilterModel.getData();
            const pagination = this.oPaginationModel.getData();
            
            this.sendWebSocketMessage({
                type: 'get_notifications',
                status: filters.status || undefined,
                type: filters.type || undefined,
                priority: filters.priority || undefined,
                category: filters.category || undefined,
                limit: pagination.limit,
                offset: pagination.offset
            });
        },

        loadUserPreferences: function () {
            // Preferences are loaded via WebSocket auth_success message
            // This method can be used for REST API fallback if needed
        },

        updateNotifications: function (notifications) {
            this.oNotificationModel.setProperty("/notifications", notifications || []);
        },

        updateStats: function (stats) {
            if (stats) {
                this.oNotificationModel.setProperty("/stats", stats);
            }
        },

        addNotification: function (notification) {
            const notifications = this.oNotificationModel.getProperty("/notifications");
            notifications.unshift(notification);
            this.oNotificationModel.setProperty("/notifications", notifications);
            
            // Update stats
            this.loadNotifications(); // Refresh to get updated stats
        },

        markNotificationAsRead: function (notificationId) {
            const notifications = this.oNotificationModel.getProperty("/notifications");
            const notification = notifications.find(n => n.ID === notificationId);
            if (notification) {
                notification.status = 'read';
                notification.readAt = new Date().toISOString();
                this.oNotificationModel.refresh();
            }
        },

        markAllNotificationsAsRead: function () {
            const notifications = this.oNotificationModel.getProperty("/notifications");
            notifications.forEach(n => {
                if (n.status === 'unread') {
                    n.status = 'read';
                    n.readAt = new Date().toISOString();
                }
            });
            this.oNotificationModel.refresh();
        },

        removeNotification: function (notificationId) {
            const notifications = this.oNotificationModel.getProperty("/notifications");
            const index = notifications.findIndex(n => n.ID === notificationId);
            if (index >= 0) {
                notifications.splice(index, 1);
                this.oNotificationModel.refresh();
            }
        },

        showNotificationToast: function (notification) {
            const message = `${notification.title}: ${notification.message}`;
            MessageToast.show(message, {
                duration: 5000,
                width: "25em"
            });
        },

        setupAutoRefresh: function () {
            // Refresh notifications every 5 minutes as fallback
            this.autoRefreshInterval = setInterval(() => {
                if (this.oNotificationModel.getProperty("/connectionStatus") === "disconnected") {
                    this.loadNotifications();
                }
            }, 300000);
        },

        // Event Handlers

        onNavBack: function () {
            window.history.go(-1);
        },

        onRefresh: function () {
            this.loadNotifications();
            MessageToast.show("Notifications refreshed");
        },

        onOpenSettings: function () {
            this.openSettingsDialog();
        },

        onFilterChange: function () {
            this.oPaginationModel.setProperty("/offset", 0);
            this.oPaginationModel.setProperty("/currentPage", 1);
            this.loadNotifications();
        },

        onClearFilters: function () {
            this.oFilterModel.setData({
                status: "",
                type: "",
                priority: "",
                category: ""
            });
            this.loadNotifications();
        },

        onNotificationPress: function (oEvent) {
            const oContext = oEvent.getSource().getBindingContext("notificationModel");
            const notification = oContext.getObject();
            
            // Mark as read if unread
            if (notification.status === 'unread') {
                this.markNotificationRead(notification.ID);
            }
        },

        onNotificationSelect: function (oEvent) {
            // Handle selection if needed
        },

        onMarkRead: function (oEvent) {
            const oContext = oEvent.getSource().getBindingContext("notificationModel");
            const notification = oContext.getObject();
            this.markNotificationRead(notification.ID);
        },

        onMarkAllRead: function () {
            MessageBox.confirm("Mark all notifications as read?", {
                onOK: () => {
                    this.sendWebSocketMessage({
                        type: 'mark_all_read'
                    });
                }
            });
        },

        onDismiss: function (oEvent) {
            const oContext = oEvent.getSource().getBindingContext("notificationModel");
            const notification = oContext.getObject();
            this.dismissNotification(notification.ID);
        },

        onDelete: function (oEvent) {
            const oContext = oEvent.getSource().getBindingContext("notificationModel");
            const notification = oContext.getObject();
            
            MessageBox.confirm("Delete this notification?", {
                onOK: () => {
                    this.deleteNotification(notification.ID);
                }
            });
        },

        onActionPress: function (oEvent) {
            const actionType = oEvent.getSource().data("actionType");
            const target = oEvent.getSource().data("target");
            
            switch (actionType) {
                case 'navigate':
                    this.navigateToTarget(target);
                    break;
                case 'api_call':
                    this.callAPI(target);
                    break;
                case 'external_link':
                    window.open(target, '_blank');
                    break;
            }
        },

        onPreviousPage: function () {
            const pagination = this.oPaginationModel.getData();
            if (pagination.offset > 0) {
                pagination.offset -= pagination.limit;
                pagination.currentPage--;
                this.oPaginationModel.refresh();
                this.loadNotifications();
            }
        },

        onNextPage: function () {
            const pagination = this.oPaginationModel.getData();
            pagination.offset += pagination.limit;
            pagination.currentPage++;
            this.oPaginationModel.refresh();
            this.loadNotifications();
        },

        // Settings Dialog

        openSettingsDialog: function () {
            if (!this.settingsDialog) {
                Fragment.load({
                    name: "a2a.network.view.fragment.NotificationSettings",
                    controller: this
                }).then((oDialog) => {
                    this.settingsDialog = oDialog;
                    this.getView().addDependent(oDialog);
                    
                    // Load current settings
                    const currentSettings = this.oSettingsModel.getData() || {};
                    this.settingsDialog.getModel("settingsModel").setData(currentSettings);
                    
                    oDialog.open();
                });
            } else {
                this.settingsDialog.open();
            }
        },

        onSaveSettings: function () {
            const settings = this.oSettingsModel.getData();
            
            this.sendWebSocketMessage({
                type: 'update_preferences',
                preferences: settings
            });
            
            this.settingsDialog.close();
        },

        onCancelSettings: function () {
            this.settingsDialog.close();
        },

        onEnablePush: function () {
            if ('serviceWorker' in navigator && 'PushManager' in window) {
                this.registerServiceWorker().then(() => {
                    return this.subscribeToPush();
                }).then((subscription) => {
                    this.oSettingsModel.setProperty("/pushToken", subscription.endpoint);
                    MessageToast.show("Push notifications enabled");
                }).catch((error) => {
                    console.error("Failed to enable push notifications:", error);
                    MessageToast.show("Failed to enable push notifications");
                });
            } else {
                MessageToast.show("Push notifications not supported");
            }
        },

        onDisablePush: function () {
            this.oSettingsModel.setProperty("/pushToken", null);
            MessageToast.show("Push notifications disabled");
        },

        // Helper Methods

        markNotificationRead: function (notificationId) {
            this.sendWebSocketMessage({
                type: 'mark_read',
                notificationId: notificationId
            });
        },

        dismissNotification: function (notificationId) {
            this.sendWebSocketMessage({
                type: 'dismiss',
                notificationId: notificationId
            });
        },

        deleteNotification: function (notificationId) {
            // For now, treat delete as dismiss
            this.dismissNotification(notificationId);
        },

        navigateToTarget: function (target) {
            // Implement navigation based on your router
            console.log("Navigate to:", target);
        },

        callAPI: function (endpoint) {
            // Implement API call
            console.log("Call API:", endpoint);
        },

        registerServiceWorker: function () {
            return navigator.serviceWorker.register('/sw.js');
        },

        subscribeToPush: function () {
            return navigator.serviceWorker.ready.then((registration) => {
                return registration.pushManager.subscribe({
                    userVisibleOnly: true,
                    applicationServerKey: this.urlBase64ToUint8Array(this.getVapidPublicKey())
                });
            });
        },

        getVapidPublicKey: function () {
            // Return your VAPID public key
            return 'your-vapid-public-key-here';
        },

        urlBase64ToUint8Array: function (base64String) {
            const padding = '='.repeat((4 - base64String.length % 4) % 4);
            const base64 = (base64String + padding)
                .replace(/-/g, '+')
                .replace(/_/g, '/');

            const rawData = window.atob(base64);
            const outputArray = new Uint8Array(rawData.length);

            for (let i = 0; i < rawData.length; ++i) {
                outputArray[i] = rawData.charCodeAt(i);
            }
            return outputArray;
        },

        // Formatters

        formatNotificationIcon: function (type, priority) {
            const iconMap = {
                'info': 'sap-icon://information',
                'warning': 'sap-icon://warning',
                'error': 'sap-icon://error',
                'success': 'sap-icon://sys-enter-2',
                'system': 'sap-icon://system-exit-2'
            };
            
            return iconMap[type] || 'sap-icon://notification-2';
        },

        formatNotificationColor: function (type, priority) {
            if (priority === 'critical') return 'Critical';
            
            const colorMap = {
                'info': 'Information',
                'warning': 'Warning',
                'error': 'Error',
                'success': 'Success',
                'system': 'Information'
            };
            
            return colorMap[type] || 'Information';
        },

        formatPriorityState: function (priority) {
            const stateMap = {
                'low': 'None',
                'medium': 'Information',
                'high': 'Warning',
                'critical': 'Error'
            };
            
            return stateMap[priority] || 'None';
        },

        formatActionButtonType: function (style) {
            const typeMap = {
                'default': 'Default',
                'primary': 'Emphasized',
                'success': 'Accept',
                'warning': 'Attention',
                'danger': 'Reject'
            };
            
            return typeMap[style] || 'Default';
        },

        // Cleanup

        onExit: function () {
            if (this.ws) {
                this.ws.close();
            }
            if (this.autoRefreshInterval) {
                clearInterval(this.autoRefreshInterval);
            }
        }
    });
});