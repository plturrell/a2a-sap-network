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
            this._checkUserAuthorization().then(function() {
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

                // Initialize AI-powered personalization
                this._initializePersonalizationAI();
                
                this._initializeDataConnection();
                this._setupConnectionHealthCheck();
            }.bind(this)).catch(function(error) {
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
            
            const handleReconnectTimeout = function() {
                this._reconnectTimer = null;
                this._initializeWebSocket();
            }.bind(this);
            this._reconnectTimer = setTimeout(handleReconnectTimeout, delay);
        },

        _clearReconnectTimer: function() {
            if (this._reconnectTimer) {
                clearTimeout(this._reconnectTimer);
                this._reconnectTimer = null;
            }
        },

        _setupConnectionHealthCheck: function() {
            const performConnectionHealthCheck = function() {
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
            }.bind(this);
            this._connectionHealthCheckInterval = setInterval(performConnectionHealthCheck, 30000);
        },

        _fetchAndSetTileData: function () {
            const handleTileDataSuccess = (data) => {
                this._updateModelWithData(data);
            };
            const handleTileDataError = (error) => {
                Log.error("Failed to fetch tile data", error);
                const fallbackData = { agentCount: 9, services: 0, workflows: 0, performance: 85, notifications: 3, security: 0 };
                this._updateModelWithData(fallbackData);
            };
            this._securityService.secureAjax({
                url: '/api/v1/Agents?id=agent_visualization',
                method: 'GET',
                dataType: 'json'
            }).then(handleTileDataSuccess).catch(handleTileDataError);
        },

        _updateModelWithData: function(data) {
            const oModel = this.getView().getModel("launchpad");
            const aTiles = oModel.getProperty("/tiles");
            const updateTileValue = (oTile) => {
                oTile.value = data[oTile.info] || 0;
            };
            aTiles.forEach(updateTileValue);
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
            // Initialize AI-powered personalization data
            const userProfile = this._getUserPersonalizationProfile();
            const personalizationModel = new JSONModel({
                selectedTheme: userProfile.preferredTheme || sap.ui.getCore().getConfiguration().getTheme(),
                contentDensity: userProfile.preferredDensity || "cozy",
                selectedLanguage: userProfile.preferredLanguage || sap.ui.getCore().getConfiguration().getLanguage(),
                dashboardLayout: userProfile.dashboardLayout || "default",
                widgetPreferences: userProfile.widgetPreferences || {},
                aiRecommendations: userProfile.aiRecommendations || {}
            });
            this.getView().setModel(personalizationModel, "personalization");
            
            if (!this._oPersonalizationDialog) {
                const handlePersonalizationDialogLoad = function (oDialog) {
                    this.getView().addDependent(oDialog);
                    return oDialog;
                }.bind(this);
                this._oPersonalizationDialog = Fragment.load({
                    name: "a2a.network.launchpad.view.Personalization",
                    controller: this
                }).then(handlePersonalizationDialogLoad);
            }

            const openPersonalizationDialog = function(oDialog) {
                oDialog.open();
            };
            this._oPersonalizationDialog.then(openPersonalizationDialog);
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
            
            // Apply dashboard layout
            if (personalizationData.dashboardLayout) {
                this._applyDashboardLayout(personalizationData.dashboardLayout);
            }
            
            // Apply widget preferences
            if (personalizationData.widgetPreferences) {
                this._applyWidgetPreferences(personalizationData.widgetPreferences);
            }
            
            // Save personalization profile
            this._savePersonalizationProfile(personalizationData);
            
            // Record user interaction for AI learning
            this._recordPersonalizationInteraction(personalizationData);
            
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
                const handleNotificationsPopoverLoad = function (oPopover) {
                    this.getView().addDependent(oPopover);
                    return oPopover;
                }.bind(this);
                this._oNotificationsPopover = Fragment.load({
                    name: "a2a.network.launchpad.view.NotificationCenter",
                    controller: this
                }).then(handleNotificationsPopoverLoad);
            }
            const openNotificationsPopover = function (oPopover) {
                oPopover.openBy(oEvent.getSource());
            };
            this._oNotificationsPopover.then(openNotificationsPopover);
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
            const markNotificationAsRead = (notification) => {
                notification.read = true;
            };
            notifications.forEach(markNotificationAsRead);
            this.getView().getModel("notifications").setProperty("/items", notifications);
            this._updateNotificationCounts();
        },

        onClearAllNotifications: function() {
            this.showStandardConfirmation({
                message: "Are you sure you want to clear all notifications?",
                title: "Clear Notifications",
                onConfirm: function() {
                    this.getView().getModel("notifications").setProperty("/items", []);
                    this._updateNotificationCounts();
                    this.handleStandardSuccess("All notifications cleared");
                }.bind(this)
            });
        },

        onNotificationSettings: function() {
            this.showStatusMessage("Notification settings not yet implemented", "Information");
        },

        onCloseNotifications: function() {
            if (this._oNotificationsPopover) {
                const closeNotificationsPopover = (popover) => popover.close();
                this._oNotificationsPopover.then(closeNotificationsPopover);
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
                const clearIntervalById = function(intervalId) {
                    clearInterval(intervalId);
                };
                this._intervals.forEach(clearIntervalById);
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
            
            // Clear personalization tracking
            if (this._timeTrackingInterval) {
                clearInterval(this._timeTrackingInterval);
                this._timeTrackingInterval = null;
            }
        },
        
        // AI-Powered Personalization Functions
        _initializePersonalizationAI: function() {
            // Initialize user behavior tracking
            this._userBehavior = {
                interactions: [],
                preferences: {},
                sessionStart: Date.now(),
                tileClicks: {},
                timeSpent: {},
                navigationPatterns: []
            };
            
            // Load existing personalization profile
            this._loadPersonalizationProfile();
            
            // Start behavior tracking
            this._startBehaviorTracking();
            
            // Apply existing personalization
            this._applyStoredPersonalization();
        },
        
        _getUserPersonalizationProfile: function() {
            const userId = this._getCurrentUserId();
            const storedProfile = localStorage.getItem(`a2a_personalization_${userId}`);
            
            if (storedProfile) {
                return JSON.parse(storedProfile);
            }
            
            // Generate AI recommendations for new users
            return this._generateInitialRecommendations();
        },
        
        _generateInitialRecommendations: function() {
            // AI-powered initial recommendations based on user role, time of day, etc.
            const currentHour = new Date().getHours();
            const userRole = this._getUserRole();
            
            return {
                preferredTheme: currentHour > 18 || currentHour < 6 ? "sap_horizon_dark" : "sap_horizon",
                preferredDensity: "cozy",
                dashboardLayout: userRole === "admin" ? "detailed" : "simplified",
                widgetPreferences: {
                    priorityWidgets: ["agentCount", "performance", "notifications"],
                    hiddenWidgets: [],
                    customOrder: []
                },
                aiRecommendations: {
                    suggestedTheme: "Based on time of day preferences",
                    suggestedLayout: "Optimized for your role",
                    suggestedWidgets: "Most relevant for your workflow"
                }
            };
        },
        
        _startBehaviorTracking: function() {
            // Track tile interactions
            const tiles = this.byId("tileContainer");
            if (tiles) {
                tiles.attachPress(function(event) {
                    const tileInfo = event.getSource().data("info");
                    this._recordTileInteraction(tileInfo);
                }.bind(this));
            }
            
            // Track time spent on different sections
            this._startTimeTracking();
            
            // Track navigation patterns
            this._trackNavigationPatterns();
        },
        
        _recordTileInteraction: function(tileInfo) {
            const timestamp = Date.now();
            
            // Update tile click count
            this._userBehavior.tileClicks[tileInfo] = 
                (this._userBehavior.tileClicks[tileInfo] || 0) + 1;
            
            // Record interaction
            this._userBehavior.interactions.push({
                type: "tile_click",
                target: tileInfo,
                timestamp: timestamp,
                sessionTime: timestamp - this._userBehavior.sessionStart
            });
            
            // Trigger personalization update if enough data
            if (this._userBehavior.interactions.length > 0 && 
                this._userBehavior.interactions.length % 10 === 0) {
                this._updatePersonalizationRecommendations();
            }
        },
        
        _startTimeTracking: function() {
            this._timeTrackingInterval = setInterval(function() {
                // Track time spent in current view
                const currentView = "launchpad";
                this._userBehavior.timeSpent[currentView] = 
                    (this._userBehavior.timeSpent[currentView] || 0) + 5000; // 5 seconds
            }.bind(this), 5000);
        },
        
        _trackNavigationPatterns: function() {
            const router = this.getOwnerComponent().getRouter();
            router.attachRouteMatched(function(event) {
                const routeName = event.getParameter("name");
                this._userBehavior.navigationPatterns.push({
                    route: routeName,
                    timestamp: Date.now()
                });
            }.bind(this));
        },
        
        _updatePersonalizationRecommendations: function() {
            // Analyze user behavior patterns
            const recommendations = this._analyzeUserBehavior();
            
            // Update personalization model
            const personalModel = this.getView().getModel("personalization");
            if (personalModel) {
                personalModel.setProperty("/aiRecommendations", recommendations);
            }
            
            // Show recommendations to user (non-intrusive)
            this._showPersonalizationSuggestions(recommendations);
        },
        
        _analyzeUserBehavior: function() {
            const behavior = this._userBehavior;
            const recommendations = {};
            
            // Analyze tile usage patterns
            const mostUsedTiles = Object.entries(behavior.tileClicks)
                .sort(([,a], [,b]) => b - a)
                .slice(0, 3)
                .map(([tile]) => tile);
            
            recommendations.priorityWidgets = mostUsedTiles;
            
            // Analyze time of day preferences
            const interactionTimes = behavior.interactions.map(i => 
                new Date(i.timestamp).getHours()
            );
            
            if (interactionTimes.length > 0) {
                const avgHour = interactionTimes.reduce((a, b) => a + b, 0) / interactionTimes.length;
                
                if (avgHour > 18 || avgHour < 6) {
                    recommendations.suggestedTheme = "sap_horizon_dark";
                    recommendations.themeReason = "You seem to use the system during evening hours";
                }
            }
            
            // Analyze interaction frequency
            const avgInteractionGap = this._calculateAvgInteractionGap();
            if (avgInteractionGap < 2000) { // Very frequent interactions
                recommendations.suggestedDensity = "compact";
                recommendations.densityReason = "High interaction frequency suggests compact layout";
            }
            
            return recommendations;
        },
        
        _calculateAvgInteractionGap: function() {
            const interactions = this._userBehavior.interactions;
            if (interactions.length < 2) return 0;
            
            const gaps = [];
            for (let i = 1; i < interactions.length; i++) {
                gaps.push(interactions[i].timestamp - interactions[i-1].timestamp);
            }
            
            return gaps.reduce((a, b) => a + b, 0) / gaps.length;
        },
        
        _showPersonalizationSuggestions: function(recommendations) {
            // Show subtle, non-intrusive suggestions
            if (recommendations.suggestedTheme && 
                recommendations.suggestedTheme !== this._getCurrentTheme()) {
                
                // Show a discrete notification about theme recommendation
                setTimeout(function() {
                    MessageToast.show(
                        "AI Suggestion: " + (recommendations.themeReason || "Consider switching to a different theme"),
                        { duration: 3000 }
                    );
                }, 5000);
            }
        },
        
        _applyDashboardLayout: function(layoutType) {
            // Apply different dashboard layouts based on AI recommendations
            const tileContainer = this.byId("tileContainer");
            if (!tileContainer) return;
            
            switch (layoutType) {
                case "detailed":
                    // Show all tiles with detailed information
                    this._showDetailedLayout(tileContainer);
                    break;
                case "simplified":
                    // Show only most important tiles
                    this._showSimplifiedLayout(tileContainer);
                    break;
                case "customized":
                    // Apply user-specific customizations
                    this._showCustomizedLayout(tileContainer);
                    break;
                default:
                    // Default layout
                    break;
            }
        },
        
        _showDetailedLayout: function(tileContainer) {
            // Show all tiles with expanded information (would need access to tiles)
            Log.info("Applied detailed dashboard layout");
        },
        
        _showSimplifiedLayout: function(tileContainer) {
            // Hide less important tiles, show key ones in compact form
            const priorityTiles = this._userBehavior.tileClicks ? 
                Object.keys(this._userBehavior.tileClicks) : [];
            
            Log.info("Applied simplified dashboard layout with priority tiles: " + priorityTiles.join(", "));
        },
        
        _applyWidgetPreferences: function(preferences) {
            // Apply widget-specific preferences
            if (preferences.hiddenWidgets && preferences.hiddenWidgets.length > 0) {
                Log.info("Hidden widgets: " + preferences.hiddenWidgets.join(", "));
            }
            
            if (preferences.priorityWidgets && preferences.priorityWidgets.length > 0) {
                Log.info("Priority widgets: " + preferences.priorityWidgets.join(", "));
            }
        },
        
        _savePersonalizationProfile: function(personalizationData) {
            const userId = this._getCurrentUserId();
            const profile = {
                ...personalizationData,
                lastUpdated: Date.now(),
                userBehavior: this._userBehavior
            };
            
            localStorage.setItem(`a2a_personalization_${userId}`, JSON.stringify(profile));
            
            // Also send to backend for cross-device sync
            this._syncPersonalizationToBackend(profile);
        },
        
        _loadPersonalizationProfile: function() {
            const userId = this._getCurrentUserId();
            const storedProfile = localStorage.getItem(`a2a_personalization_${userId}`);
            
            if (storedProfile) {
                const profile = JSON.parse(storedProfile);
                if (profile.userBehavior) {
                    this._userBehavior = {...this._userBehavior, ...profile.userBehavior};
                }
            }
        },
        
        _applyStoredPersonalization: function() {
            const profile = this._getUserPersonalizationProfile();
            
            // Apply theme
            if (profile.preferredTheme && 
                profile.preferredTheme !== sap.ui.getCore().getConfiguration().getTheme()) {
                sap.ui.getCore().applyTheme(profile.preferredTheme);
            }
            
            // Apply density
            if (profile.preferredDensity) {
                document.body.classList.toggle("sapUiSizeCompact", profile.preferredDensity === "compact");
                document.body.classList.toggle("sapUiSizeCozy", profile.preferredDensity === "cozy");
            }
            
            // Apply layout
            if (profile.dashboardLayout) {
                setTimeout(function() {
                    this._applyDashboardLayout(profile.dashboardLayout);
                }.bind(this), 1000);
            }
        },
        
        _recordPersonalizationInteraction: function(personalizationData) {
            // Record this personalization change for AI learning
            this._userBehavior.interactions.push({
                type: "personalization_change",
                changes: personalizationData,
                timestamp: Date.now(),
                sessionTime: Date.now() - this._userBehavior.sessionStart
            });
        },
        
        _syncPersonalizationToBackend: function(profile) {
            // Send personalization data to backend for analysis and cross-device sync
            this._securityService.secureAjax({
                url: '/api/v1/personalization/sync',
                method: 'POST',
                data: JSON.stringify({
                    userId: this._getCurrentUserId(),
                    profile: profile,
                    timestamp: Date.now()
                }),
                contentType: 'application/json'
            }).catch(function(error) {
                Log.warning("Failed to sync personalization to backend", error);
            });
        },
        
        _getCurrentUserId: function() {
            // Get current user ID (simplified)
            return "user_" + (new Date().getTime() % 10000);
        },
        
        _getCurrentTheme: function() {
            return sap.ui.getCore().getConfiguration().getTheme();
        },
        
        _getUserRole: function() {
            // Determine user role (simplified)
            return "user"; // Would be determined from authentication context
        }
    }));
});
