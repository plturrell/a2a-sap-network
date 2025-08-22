/* global sap */
sap.ui.define([
    "sap/ui/core/mvc/Controller",
    "sap/ui/model/json/JSONModel",
    "sap/ui/core/Fragment",
    "sap/m/MessageToast",
    "sap/base/Log",
    "a2a/network/launchpad/services/SecurityService",
    "a2a/network/launchpad/controller/mixin/StandardPatternsMixin"
], function (Controller, JSONModel, Fragment, MessageToast, Log, SecurityService, StandardPatternsMixin) {
    "use strict";

    return Controller.extend("a2a.network.launchpad.controller.Launchpad", Object.assign({}, StandardPatternsMixin, {
        _intervals: [],
        _websocket: null,
        _reconnectAttempts: 0,
        _maxReconnectAttempts: 10,
        _reconnectDelay: 1000,
        _reconnectTimer: null,
        _connectionHealthCheckInterval: null,
        _lastDataUpdate: null,

        onInit: function () {
            // Initialize standard patterns
            this.initializeStandardPatterns();
            // Initialize security service
            this._securityService = new SecurityService();
            
            // Initialize system status model
            this._initializeSystemModels();
            
            // Check authorization before proceeding
            this._checkUserAuthorization().then(() => {
                const oI18nModel = this.getOwnerComponent().getModel("i18n");
                const oResourceBundle = oI18nModel.getResourceBundle();

            const oModel = new JSONModel({
                tiles: [
                    { header: oResourceBundle.getText("agentManagementTileHeader"), subheader: oResourceBundle.getText("agentManagementTileSubheader"), icon: "sap-icon://collaborate", value: 0, info: "agentCount" },
                    { header: oResourceBundle.getText("serviceMarketplaceTileHeader"), subheader: oResourceBundle.getText("serviceMarketplaceTileSubheader"), icon: "sap-icon://sales-order", value: 0, info: "services" },
                    { header: oResourceBundle.getText("workflowDesignerTileHeader"), subheader: oResourceBundle.getText("workflowDesignerTileSubheader"), icon: "sap-icon://workflow-tasks", value: 0, info: "workflows" },
                    { header: oResourceBundle.getText("networkAnalyticsTileHeader"), subheader: oResourceBundle.getText("networkAnalyticsTileSubheader"), icon: "sap-icon://business-objects-experience", value: 0, info: "performance" },
                    { header: oResourceBundle.getText("notificationCenterTileHeader"), subheader: oResourceBundle.getText("notificationCenterTileSubheader"), icon: "sap-icon://bell", value: 0, info: "notifications" },
                    { header: oResourceBundle.getText("securityAuditTileHeader"), subheader: oResourceBundle.getText("securityAuditTileSubheader"), icon: "sap-icon://shield", value: 0, info: "security" }
                ]
            });
            this.getView().setModel(oModel, "launchpad");

                this._initializeDataConnection();
                this._setupConnectionHealthCheck();
            }).catch(error => {
                Log.error("Authorization check failed", error);
                MessageToast.show(this.getOwnerComponent().getModel("i18n").getResourceBundle().getText("authorizationError") || "You are not authorized to access this application");
            });
        },

        _initializeSystemModels: function() {
            // System status model
            const systemModel = new JSONModel({
                healthStatus: "healthy",
                activeAgents: 9,
                totalAgents: 16,
                performanceScore: 85,
                alertCount: 0
            });
            this.getView().setModel(systemModel, "system");
            
            // Notifications model with enhanced structure
            const notificationsModel = new JSONModel({
                items: [],
                itemsCount: 0,
                unreadCount: 0,
                hasUnread: false
            });
            this.getView().setModel(notificationsModel, "notifications");
            
            // User model
            const userModel = new JSONModel({
                name: "User",
                email: "",
                role: "",
                isAuthenticated: false
            });
            this.getView().setModel(userModel, "user");
            
            // Update page state for launchpad
            this.updatePageState({
                pageTitle: "A2A Network Launchpad",
                headerExpanded: true,
                searchVisible: false,
                viewControlsVisible: false,
                filterBarVisible: false
            });
        },

        _checkUserAuthorization: function() {
            // Check required authorization objects
            return Promise.all([
                this._securityService.checkAuthorization("S_SERVICE", "SRV_NAME", "ZFIORI_LAUNCHPAD"),
                this._securityService.checkAuthorization("/UI2/CHIP", "CHIP_ID", "*")
            ]).then(results => {
                if (!results[0] || !results[1]) {
                    throw new Error("User lacks required authorizations");
                }
                Log.info("User authorization check passed");
            });
        },

        _initializeDataConnection: function() {
            this._fetchAndSetTileData();
            this._intervals.push(setInterval(this._fetchAndSetTileData.bind(this), 30000));
            this._initializeWebSocket();
        },

        _initializeWebSocket: function() {
            try {
                const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
                const host = window.location.host;
                
                // Get authentication token if available
                const token = this._getAuthToken();
                
                // In development mode, WebSocket might work without token
                let wsUrl = `${protocol}//${host}/ws`;
                if (token) {
                    wsUrl += `?token=${encodeURIComponent(token)}`;
                } else {
                    Log.info("Connecting to WebSocket without authentication token (development mode)");
                }
                
                this._websocket = new WebSocket(wsUrl);
                this._setupWebSocketHandlers();
                
            } catch (error) {
                Log.error("Failed to initialize WebSocket", error);
                this._scheduleReconnect();
            }
        },

        _getAuthToken: function() {
            // Use security service for authentication token
            return this._securityService.getAuthToken();
        },

        _setupWebSocketHandlers: function() {
            if (!this._websocket) return;

            this._websocket.onopen = function() {
                Log.info("WebSocket connected successfully");
                this._reconnectAttempts = 0;
                this._clearReconnectTimer();
                
                if (this._websocket.readyState === WebSocket.OPEN) {
                    const token = this._getAuthToken();
                    this._websocket.send(JSON.stringify({ 
                        type: "subscribe", 
                        topics: ["agents", "tiles", "notifications"],
                        auth: token
                    }));
                }
                
                MessageToast.show(this.getOwnerComponent().getModel("i18n").getResourceBundle().getText("websocketConnected") || "Real-time updates connected");
            }.bind(this);

            this._websocket.onmessage = function(event) {
                try {
                    const data = JSON.parse(event.data);
                    this._handleWebSocketMessage(data);
                    this._lastDataUpdate = new Date();
                } catch (error) {
                    Log.error("Failed to parse WebSocket message", error);
                }
            }.bind(this);

            this._websocket.onerror = function(error) {
                Log.error("WebSocket error occurred", error);
            }.bind(this);

            this._websocket.onclose = function(event) {
                Log.warning("WebSocket connection closed", { code: event.code, reason: event.reason });
                this._websocket = null;
                
                if (!event.wasClean) {
                    this._scheduleReconnect();
                }
            }.bind(this);
        },

        _handleWebSocketMessage: function(data) {
            switch(data.type) {
                case "tileUpdate":
                    this._updateModelWithData(data.payload);
                    break;
                case "notification":
                    this._addNotification(data.payload);
                    break;
                case "heartbeat":
                    if (this._websocket && this._websocket.readyState === WebSocket.OPEN) {
                        this._websocket.send(JSON.stringify({ type: "pong" }));
                    }
                    break;
                default:
                    Log.debug("Received unknown WebSocket message type", data.type);
            }
        },

        _scheduleReconnect: function() {
            if (this._reconnectTimer) return;
            
            if (this._reconnectAttempts >= this._maxReconnectAttempts) {
                Log.error("Maximum WebSocket reconnection attempts reached");
                MessageToast.show(this.getOwnerComponent().getModel("i18n").getResourceBundle().getText("websocketReconnectFailed") || "Real-time connection failed. Using periodic updates.");
                return;
            }
            
            this._reconnectAttempts++;
            const delay = Math.min(this._reconnectDelay * Math.pow(2, this._reconnectAttempts - 1), 30000);
            
            Log.info(`Scheduling WebSocket reconnection attempt ${this._reconnectAttempts} in ${delay}ms`);
            
            this._reconnectTimer = setTimeout(function() {
                this._reconnectTimer = null;
                this._initializeWebSocket();
            }.bind(this), delay);
        },

        _clearReconnectTimer: function() {
            if (this._reconnectTimer) {
                clearTimeout(this._reconnectTimer);
                this._reconnectTimer = null;
            }
        },

        _setupConnectionHealthCheck: function() {
            this._connectionHealthCheckInterval = setInterval(function() {
                if (this._websocket && this._websocket.readyState === WebSocket.OPEN) {
                    const timeSinceLastUpdate = this._lastDataUpdate ? 
                        (new Date() - this._lastDataUpdate) / 1000 : Infinity;
                    
                    if (timeSinceLastUpdate > 60) {
                        Log.warning("No WebSocket data received for 60 seconds, sending ping");
                        try {
                            this._websocket.send(JSON.stringify({ type: "ping" }));
                        } catch (error) {
                            Log.error("Failed to send ping", error);
                            this._websocket.close();
                        }
                    }
                } else if (!this._websocket && this._reconnectAttempts < this._maxReconnectAttempts) {
                    Log.info("WebSocket not connected, attempting to reconnect");
                    this._scheduleReconnect();
                }
            }.bind(this), 30000);
        },

        _fetchAndSetTileData: function () {
            this._securityService.secureAjax({
                url: '/api/v1/Agents?id=agent_visualization',
                method: 'GET',
                dataType: 'json'
            }).then(data => {
                this._updateModelWithData(data);
            }).catch(error => {
                Log.error("Failed to fetch tile data", error);
                const fallbackData = { agentCount: 9, services: 0, workflows: 0, performance: 85, notifications: 3, security: 0 };
                this._updateModelWithData(fallbackData);
            });
        },

        _updateModelWithData: function(data) {
            const oModel = this.getView().getModel("launchpad");
            const aTiles = oModel.getProperty("/tiles");
            aTiles.forEach(oTile => {
                oTile.value = data[oTile.info] || 0;
            });
            oModel.setProperty("/tiles", aTiles);
        },

        _addNotification: function(notification) {
            const oNotificationsModel = this.getView().getModel("notifications");
            const aItems = oNotificationsModel.getProperty("/items") || [];
            
            aItems.unshift({
                title: notification.title || "System Notification",
                description: notification.description || notification.message,
                icon: notification.icon || "sap-icon://message-information",
                timestamp: new Date()
            });
            
            if (aItems.length > 10) {
                aItems.length = 10;
            }
            
            oNotificationsModel.setProperty("/items", aItems);
            
            const oTilesModel = this.getView().getModel("launchpad");
            const aTiles = oTilesModel.getProperty("/tiles");
            const notificationTile = aTiles.find(tile => tile.info === "notifications");
            if (notificationTile) {
                notificationTile.value = aItems.length;
                oTilesModel.setProperty("/tiles", aTiles);
            }
        },

        onOpenPersonalization: function () {
            // Initialize personalization data
            const personalizationModel = new JSONModel({
                selectedTheme: sap.ui.getCore().getConfiguration().getTheme(),
                contentDensity: "cozy",
                selectedLanguage: sap.ui.getCore().getConfiguration().getLanguage()
            });
            this.getView().setModel(personalizationModel, "personalization");
            
            if (!this._oPersonalizationDialog) {
                this._oPersonalizationDialog = Fragment.load({
                    name: "a2a.network.launchpad.view.Personalization",
                    controller: this
                }).then(function (oDialog) {
                    this.getView().addDependent(oDialog);
                    return oDialog;
                }.bind(this));
            }

            this._oPersonalizationDialog.then(function(oDialog) {
                oDialog.open();
            });
        },

        onApplyPersonalization: function() {
            const personalizationData = this.getView().getModel("personalization").getData();
            
            // Apply theme
            if (personalizationData.selectedTheme) {
                sap.ui.getCore().applyTheme(personalizationData.selectedTheme);
            }
            
            // Apply content density
            if (personalizationData.contentDensity) {
                document.body.classList.toggle("sapUiSizeCompact", personalizationData.contentDensity === "compact");
                document.body.classList.toggle("sapUiSizeCozy", personalizationData.contentDensity === "cozy");
            }
            
            this.handleStandardSuccess("Personalization settings applied successfully");
            this.onClosePersonalization();
        },

        onContentDensityChange: function(event) {
            const density = event.getParameter("item").getKey();
            this.getView().getModel("personalization").setProperty("/contentDensity", density);
        },

        onLanguageChange: function(event) {
            const language = event.getParameter("selectedItem").getKey();
            this.getView().getModel("personalization").setProperty("/selectedLanguage", language);
        },

        onOpenNotifications: function (oEvent) {
            // Sample dynamic notifications
            const oNotificationsModel = this.getView().getModel("notifications");
            oNotificationsModel.setData({
                items: [
                    { title: "System Update", description: "A new system update is available.", icon: "sap-icon://message-information" },
                    { title: "Agent Offline", description: "Agent 'Alpha-7' has gone offline.", icon: "sap-icon://message-warning" },
                    { title: "High-priority Alert", description: "Unusual network activity detected.", icon: "sap-icon://message-error" }
                ]
            });

            if (!this._oNotificationsPopover) {
                this._oNotificationsPopover = Fragment.load({
                    name: "a2a.network.launchpad.view.NotificationCenter",
                    controller: this
                }).then(function (oPopover) {
                    this.getView().addDependent(oPopover);
                    return oPopover;
                }.bind(this));
            }
            this._oNotificationsPopover.then(function (oPopover) {
                oPopover.openBy(oEvent.getSource());
            });
        },

        onClosePersonalization: function () {
            this._oPersonalizationDialog.close();
        },

        onThemeChange: function(oEvent) {
            const sTheme = oEvent.getParameter("selectedItem").getKey();
            sap.ui.getCore().applyTheme(sTheme);
        },

        onRefresh: function () {
            this._fetchAndSetTileData();
            MessageToast.show(this.getOwnerComponent().getModel("i18n").getResourceBundle().getText("refreshSuccessToast"));
        },

        onOpenAnalytics: function () {
            const oRouter = this.getOwnerComponent().getRouter();
            oRouter.navTo("Analytics");
        },

        onToggleFullScreen: function() {
            const launchpadPage = this.byId("launchpadPage");
            const isFullScreen = document.fullscreenElement;
            
            if (!isFullScreen) {
                launchpadPage.getDomRef().requestFullscreen();
            } else {
                document.exitFullscreen();
            }
        },

        onTilePress: function(event) {
            const tile = event.getSource();
            const agentId = tile.data("agentId");
            const status = tile.data("status");
            
            if (status === "active") {
                this._navigateToAgent(agentId);
            } else {
                this.showStatusMessage(`Agent ${agentId} is currently ${status}`, "Warning");
            }
        },

        onCreateAgent: function() {
            this.openStandardDialog({
                title: "Create New Agent",
                formFragment: "a2a.network.launchpad.fragment.CreateAgentForm",
                primaryButtonText: "Create",
                primaryButtonPress: "onCreateAgentConfirm"
            });
        },

        onDeployWorkflow: function() {
            this.openStandardDialog({
                title: "Deploy Workflow",
                formFragment: "a2a.network.launchpad.fragment.DeployWorkflowForm",
                primaryButtonText: "Deploy",
                primaryButtonPress: "onDeployWorkflowConfirm"
            });
        },

        onViewReports: function() {
            const oRouter = this.getOwnerComponent().getRouter();
            oRouter.navTo("Analytics", { reportType: "overview" });
        },

        onNotificationPress: function(event) {
            const bindingContext = event.getSource().getBindingContext("notifications");
            const notification = bindingContext.getObject();
            
            // Mark as read
            notification.read = true;
            this._updateNotificationCounts();
            
            // Handle notification action
            if (notification.actionUrl) {
                window.open(notification.actionUrl, "_blank");
            }
        },

        onDeleteNotification: function(event) {
            const listItem = event.getParameter("listItem");
            const bindingContext = listItem.getBindingContext("notifications");
            const notifications = this.getView().getModel("notifications").getProperty("/items");
            const index = notifications.indexOf(bindingContext.getObject());
            
            if (index > -1) {
                notifications.splice(index, 1);
                this.getView().getModel("notifications").setProperty("/items", notifications);
                this._updateNotificationCounts();
            }
        },

        onMarkAllRead: function() {
            const notifications = this.getView().getModel("notifications").getProperty("/items");
            notifications.forEach(notification => {
                notification.read = true;
            });
            this.getView().getModel("notifications").setProperty("/items", notifications);
            this._updateNotificationCounts();
        },

        onClearAllNotifications: function() {
            this.showStandardConfirmation({
                message: "Are you sure you want to clear all notifications?",
                title: "Clear Notifications",
                onConfirm: () => {
                    this.getView().getModel("notifications").setProperty("/items", []);
                    this._updateNotificationCounts();
                    this.handleStandardSuccess("All notifications cleared");
                }
            });
        },

        onNotificationSettings: function() {
            this.showStatusMessage("Notification settings not yet implemented", "Information");
        },

        onCloseNotifications: function() {
            if (this._oNotificationsPopover) {
                this._oNotificationsPopover.then(popover => popover.close());
            }
        },

        onViewAllNotifications: function() {
            const oRouter = this.getOwnerComponent().getRouter();
            oRouter.navTo("NotificationCenter");
        },

        _navigateToAgent: function(agentId) {
            const oRouter = this.getOwnerComponent().getRouter();
            oRouter.navTo("AgentDetail", { agentId: agentId });
        },

        _updateNotificationCounts: function() {
            const notifications = this.getView().getModel("notifications").getProperty("/items");
            const unreadCount = notifications.filter(n => !n.read).length;
            
            this.getView().getModel("notifications").setData({
                items: notifications,
                itemsCount: notifications.length,
                unreadCount: unreadCount,
                hasUnread: unreadCount > 0
            });
        },

        onExit: function() {
            // Clean up intervals to prevent memory leaks
            if (this._intervals) {
                this._intervals.forEach(function(intervalId) {
                    clearInterval(intervalId);
                });
                this._intervals = [];
            }
            
            // Clean up WebSocket connection
            if (this._websocket) {
                this._websocket.close(1000, "Controller destroyed");
                this._websocket = null;
            }
            
            // Clear reconnection timer
            this._clearReconnectTimer();
            
            // Clear health check interval
            if (this._connectionHealthCheckInterval) {
                clearInterval(this._connectionHealthCheckInterval);
                this._connectionHealthCheckInterval = null;
            }
        }
    });
